import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
from tqdm import tqdm

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss
    
class clientOrtho(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.op_loss = OrthogonalProjectionLoss(gamma=0.5)
        self.lamda = args.lamda

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
                loss = self.loss(output, y) + inter_orth_loss(rep, y, self.global_protos)

                # if self.global_protos is not None:
                #     loss += intra_orth_loss(rep, y, self.global_protos) * self.lamda

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

    return loss_intra

def inter_orth_loss(rep, labels, protos):
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
    rep_norm = F.normalize(rep, p=2, dim=1)   
    protos = torch.stack(list(protos.values()))       # 形状: (batch_size, feature_dim)
    protos_norm = F.normalize(protos, p=2, dim=1)   # 形状: (num_classes, feature_dim)

    # 获取每个样本对应的原型
    proto_of_sample = protos_norm[labels]            # 形状: (batch_size, feature_dim)

    # 计算类内相似度（余弦相似度）
    similarity_intra = torch.sum(rep_norm * proto_of_sample, dim=1)  # 形状: (batch_size,)
    loss_intra = 1 - similarity_intra                                # 希望最大化相似度，因此最小化 (1 - 相似度)
    loss_intra = torch.sum(loss_intra)
    # loss_intra = loss_intra / similarity_intra.shape[0]  # 计算平均损失

    # 计算与所有原型的相似度
    similarity_all = torch.matmul(rep_norm, protos_norm.t())        # 形状: (batch_size, num_classes)
    similarity_all = torch.abs(similarity_all)

    # 创建掩码，将正确类别的相似度置零
    mask = torch.ones_like(similarity_all)
    mask.scatter_(1, labels.view(-1, 1), 0)                        # 正确类别相似度置零

    # 只保留上三角部分的掩码，以避免双重计算
    upper_tri_mask = torch.triu(torch.ones_like(mask), diagonal=1)
    mask = mask * upper_tri_mask

    # 计算类间损失
    similarity_other = similarity_all * mask                        # 仅保留其他类别的相似度，并避免双重计算
    loss_inter = torch.sum(similarity_other)                        # 对所有类别求和
    # loss_inter = loss_inter / torch.sum(mask)                     # 计算平均损失

    # 合并所有损失
    loss = loss_intra + loss_inter

    return loss