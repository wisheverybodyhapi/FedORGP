import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import KMeans


class clientOrtho(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.c_lamda = args.c_lamda
        self.alpha = args.alpha

        self.global_protos = None
        self.protos = None

    def train(self):
        trainloader = self.load_train_data()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # for step in range(max_local_epochs):  # tqdm 包裹外层循环
        #     for i, (x, y) in enumerate(tqdm(trainloader, desc=f"client{self.id} Epoch {step + 1}", leave=False)):  # tqdm 包裹内层数据加载
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):  # tqdm 包裹内层数据加载
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # if self.train_slow:
                #     time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)
                if self.global_protos != None:
                    loss += self.c_lamda * intra_orth_loss(rep, y, self.global_protos)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.collect_protos()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_protos(self, global_protos):
        # receive global protos
        self.global_protos = global_protos

    def collect_protos(self):
        trainloader = self.load_train_data()

        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # if self.train_slow:
                #     time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        # 对每个类别的特征做聚类去除异常值
        for label in protos:
            protos[label] = self.filter_outliers_multi_metric(protos[label], alpha=self.alpha)

        self.protos = agg_func(protos)

    def test_metrics(self):
        testloader = self.load_test_data()

        self.model.eval()

        test_acc = 0
        test_num = 0

        if self.global_protos is not None:
            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = self.model.base(x)
                    output = self.model.head(rep)

                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    def filter_outliers_multi_metric(self, features, alpha=0.5, min_keep_ratio=0.5):
        """
        Simplified adaptive multi-metric feature filtering
        简化的自适应多维度特征筛选，减少超参数数量
        
        Args:
            features: List of feature tensors
            alpha: Weight for combining distance and cosine similarity scores (0-1) - 唯一超参数
            min_keep_ratio: Minimum ratio of features to keep (防止过度筛选) - 固定为0.5
        """
        if len(features) <= 2:
            return features
        
        feats = torch.stack(features).cpu()
        n_samples = feats.shape[0]
        
        # 使用均值作为中心点
        center = torch.mean(feats, dim=0)
        
        # Calculate Euclidean distances
        dists = torch.norm(feats - center, dim=1)
        
        # Calculate cosine similarities
        feats_norm = F.normalize(feats, p=2, dim=1)
        center_norm = F.normalize(center.unsqueeze(0), p=2, dim=1)
        cos_sims = torch.matmul(feats_norm, center_norm.t()).squeeze(1)
        
        # === 动态自适应权重计算 ===
        # 基于标准差比值的动态调整，更加科学和自适应
        dist_std = torch.std(dists)
        cos_std = torch.std(cos_sims)
        
        # 防止除零错误
        dist_std = torch.clamp(dist_std, min=1e-8)
        cos_std = torch.clamp(cos_std, min=1e-8)
        
        # 计算变异系数（标准差/均值），更公平地比较不同指标的稳定性
        dist_mean = torch.mean(dists)
        cos_mean = torch.mean(cos_sims)
        
        # 防止除零
        dist_mean = torch.clamp(dist_mean, min=1e-8)
        cos_mean = torch.clamp(torch.abs(cos_mean), min=1e-8)
        
        # 变异系数：标准差除以均值的绝对值
        dist_cv = dist_std / dist_mean
        cos_cv = cos_std / cos_mean
        
        # 基于变异系数的比值进行调整，更加公平
        cv_ratio = dist_cv / cos_cv
        
        # 改进的动态调整：使用更平滑的sigmoid函数
        # 当cv_ratio = 1时，adjustment_factor = 1（无调整）
        # 使用tanh函数实现平滑调整
        adjustment_factor = 1.0 - 0.3 * torch.tanh(0.5 * (cv_ratio - 1.0)).item()
        
        # 限制调整范围在[0.4, 1.6]之间，允许更大的调整幅度
        adjustment_factor = torch.clamp(torch.tensor(adjustment_factor), 0.4, 1.6).item()
        
        # 应用动态调整
        adaptive_alpha = alpha * adjustment_factor
        
        # 最终clamp，确保权重在合理范围内
        adaptive_alpha = torch.clamp(torch.tensor(adaptive_alpha), 0.1, 0.9).item()
        
        # === 归一化分数计算 ===
        # 归一化到[0,1]范围，距离越小分数越高，相似度越高分数越高
        dist_scores = 1 - (dists - torch.min(dists)) / (torch.max(dists) - torch.min(dists) + 1e-8)
        cos_scores = (cos_sims - torch.min(cos_sims)) / (torch.max(cos_sims) - torch.min(cos_sims) + 1e-8)
        
        # 组合分数：使用自适应alpha
        combined_scores = adaptive_alpha * dist_scores + (1 - adaptive_alpha) * cos_scores
        
        # === 简化的异常值检测 ===
        # 使用固定的IQR方法，移除复杂的条件分支
        q1 = torch.quantile(combined_scores, 0.25)
        q3 = torch.quantile(combined_scores, 0.75)
        iqr = q3 - q1
        
        # 固定IQR倍数为1.5（经典异常值检测标准）
        combined_threshold = q1 - 1.5 * iqr
        
        # === 移除冗余的basic_keep，直接使用组合分数 ===
        keep_idx = combined_scores >= combined_threshold
        
        # 防止过度筛选：确保至少保留min_keep_ratio的样本
        n_keep = torch.sum(keep_idx).item()
        min_keep = int(n_samples * min_keep_ratio)
        
        if n_keep < min_keep:
            # 如果筛选太严格，保留分数最高的min_keep个样本
            _, top_indices = torch.topk(combined_scores, min_keep)
            keep_idx = torch.zeros(n_samples, dtype=torch.bool)
            keep_idx[top_indices] = True
        
        filtered = [features[i] for i in range(len(features)) if keep_idx[i]]
        
        # 确保至少保留一个样本
        if len(filtered) == 0:
            best_idx = torch.argmax(combined_scores)
            filtered = [features[best_idx]]
    
        # 详细的调试信息
        # self.logger.write(f"Client {self.id} - Adaptive filtering debug:")
        # self.logger.write(f"  Original: {len(features)}, Filtered: {len(filtered)}, Keep ratio: {len(filtered)/len(features):.2f}")
        # self.logger.write(f"  dist_cv: {dist_cv:.4f}, cos_cv: {cos_cv:.4f}, cv_ratio: {cv_ratio:.4f}")
        # self.logger.write(f"  adjustment_factor: {adjustment_factor:.4f}, adaptive_alpha: {adaptive_alpha:.3f}")
        
        return filtered

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # if self.train_slow:
                #     time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += intra_orth_loss(rep, y, self.global_protos) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def intra_orth_loss(rep, labels, protos):
    # Calculate the orthogonal loss between the representation and the global prototype of the same class
    """
    参数：
        rep (Tensor): 表征张量，形状为 (batch_size, feature_dim)。
        labels (Tensor): 真实标签，形状为 (batch_size,)。
        protos (Tensor): 原型张量，形状为 (num_classes, feature_dim)。
    返回：
        Tensor: 计算得到的正交损失。
    """
    # 归一化表征和原型
    rep_norm = F.normalize(rep, p=2, dim=1)          # 形状: (batch_size, feature_dim)
    protos = torch.stack(list(protos.values()))
    protos_norm = F.normalize(protos, p=2, dim=1)    # 形状: (num_classes, feature_dim)

    # 获取每个样本对应的原型
    proto_of_sample = protos_norm[labels]            # 形状: (batch_size, feature_dim)

    # 计算类内相似度损失（余弦相似度）
    similarity_intra = torch.sum(rep_norm * proto_of_sample, dim=1)  # 形状: (batch_size,)
    loss_intra = 1 - similarity_intra                                # 希望最大化相似度，因此最小化 (1 - 相似度)
    loss_intra = torch.sum(loss_intra)
    loss_intra = loss_intra / similarity_intra.shape[0]  # 计算平均损失

    return loss_intra

