import time
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import itertools
from flcore.clients.clientorth import clientOrtho
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from collections import defaultdict
from torch.utils.data import DataLoader
from utils.model_utils import check_for_model
from tqdm import tqdm

class FedOrth(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientOrtho)

        self.logger.write(f"Join ratio / total clients: {self.join_ratio} / {self.num_clients}")
        self.logger.write("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs
        self.margin_threthold = args.margin_threthold

        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim
        # modification 1

        if self.save_folder_name == 'temp' or check_for_model(self.model_folder_name) == False:
            self.PROTO = Trainable_prototypes(
                self.num_classes, 
                self.server_hidden_dim, 
                self.feature_dim, 
                self.device
            ).to(self.device)
            self.logger.write(self.PROTO)
        else:
            self.PROTO = load_item(self.role, 'PROTO', self.model_folder_name)
            global_protos = defaultdict(list)
            for class_id in range(self.num_classes):
                global_protos[class_id] = self.PROTO(torch.tensor(class_id, device=self.device)).detach()
            
            self.global_protos = global_protos
            self.send_protos()
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

        self.lamda = args.lamda
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None


    def train(self):
        for i in range(self.start_round, self.end_round):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            for client in tqdm(self.selected_clients):
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()
            self.update_Gen()
            self.send_protos()
            
            if i%self.eval_gap == 0:
                self.logger.write(f"-------------Round number: {i}-------------")
                self.logger.write("Evaluate heterogeneous models")
                self.evaluate()

            self.Budget.append(time.time() - s_t)
            self.logger.write("The current global round takes {} seconds".format(self.Budget[-1]))

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        self.logger.write("Best accuracy: {}".format(max(self.rs_test_acc)))
        # self.self.logger.write_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # self.logger.write(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_models() # new modification
        
    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def save_models(self):
        # save models
        if self.save_folder_name != 'temp':
            if os.path.exists(self.model_folder_name) == False:
                os.makedirs(self.model_folder_name)
            try:
                for client in self.clients:
                    save_item(client.model, client.role, 'model', self.model_folder_name)
                self.logger.write('finish saving models of clients')
                save_item(self.PROTO, self.role, 'PROTO', self.model_folder_name)
                self.logger.write('finish saving PROTO of server')
            except Exception as e:
                self.logger.write(f"An error occurred: {str(e)}")
                self.logger.logger.exception("Exception occurred while saving models and PROTO")

    def calculate_prototype_metrics(self, uploaded_protos_per_client):
        """
        计算类间的最小欧氏距离和余弦相似度，并记录相关指标。
        """
        # 初始化gap张量，存储每个类的最小距离
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9

        # 聚合每个客户端上传的原型，得到全局平均原型
        avg_protos = proto_cluster(uploaded_protos_per_client)

        # 初始化一个字典，用于存储每对类之间的余弦相似度
        cosine_similarities = {}

        # 计算每对类之间的欧氏距离和余弦相似度
        for k1, k2 in itertools.combinations(avg_protos.keys(), 2):
            # 计算欧氏距离
            euclidean_distance = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
            # 更新每个类的最小gap
            self.gap[k1] = torch.min(self.gap[k1], euclidean_distance)
            self.gap[k2] = torch.min(self.gap[k2], euclidean_distance)

            # 计算余弦相似度
            cosine_similarity = torch.nn.functional.cosine_similarity(
                avg_protos[k1].unsqueeze(0), avg_protos[k2].unsqueeze(0)
            ).item()  # 转换为Python标量
            # 存储余弦相似度
            cosine_similarities[(k1, k2)] = cosine_similarity

        # 处理gap中未更新的类，将其设置为全局最小gap
        self.min_gap = torch.min(self.gap)
        self.gap[self.gap > 1e8] = self.min_gap

        # 计算最大gap
        self.max_gap = torch.max(self.gap)

        # 计算全局的余弦相似度统计指标
        cosine_sim_values = list(cosine_similarities.values())
        avg_cosine_similarity = sum(cosine_sim_values) / len(cosine_sim_values)
        max_cosine_similarity = max(cosine_sim_values)
        min_cosine_similarity = min(cosine_sim_values)

        # 记录欧氏距离相关指标
        self.logger.write(f'Class-wise minimum Euclidean distance: {self.gap}')
        self.logger.write(f'Minimum gap (Euclidean distance): {self.min_gap.item()}')
        self.logger.write(f'Maximum gap (Euclidean distance): {self.max_gap.item()}')

        # 记录余弦相似度相关指标
        self.logger.write(f'Average Cosine Similarity: {avg_cosine_similarity:.4f}')
        self.logger.write(f'Minimum Cosine Similarity: {min_cosine_similarity:.4f}')
        self.logger.write(f'Maximum Cosine Similarity: {max_cosine_similarity:.4f}')

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        uploaded_protos_per_client = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = client.protos
            for k in protos.keys():
                self.uploaded_protos.append((protos[k], k))
            uploaded_protos_per_client.append(protos)

        self.calculate_prototype_metrics(uploaded_protos_per_client)

    def update_Gen(self):
        Gen_opt = torch.optim.SGD(self.PROTO.parameters(), lr=self.server_learning_rate)
        self.PROTO.train()
        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, self.batch_size, 
                                      drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                y = torch.Tensor(y).type(torch.int64).to(self.device)
                proto_gen = self.PROTO(list(range(self.num_classes)))

                loss = inter_orth_loss(proto, y, proto_gen)

                Gen_opt.zero_grad()
                loss.backward()
                Gen_opt.step()

        self.logger.write(f'Server loss: {loss.item()}')
        self.uploaded_protos = []


        self.PROTO.eval()
        global_protos = defaultdict(list)
        for class_id in range(self.num_classes):
            global_protos[class_id] = self.PROTO(torch.tensor(class_id, device=self.device)).detach()
        
        self.global_protos = global_protos


def proto_cluster(protos_list):
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k in protos.keys():
            proto_clusters[k].append(protos[k])

    for k in proto_clusters.keys():
        protos = torch.stack(proto_clusters[k])
        proto_clusters[k] = torch.mean(protos, dim=0).detach()

    return proto_clusters

def inter_orth_loss(rep, labels, protos):
    # Calculate the orthogonal loss between the representation and the global prototype
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
    protos_norm = F.normalize(protos, p=2, dim=1)   # 形状: (num_classes, feature_dim)

    # 获取每个样本对应的原型
    proto_of_sample = protos_norm[labels]            # 形状: (batch_size, feature_dim)

    # 计算类内相似度（余弦相似度）
    similarity_intra = torch.sum(rep_norm * proto_of_sample, dim=1)  # 形状: (batch_size,)
    loss_intra = 1 - similarity_intra                                # 希望最大化相似度，因此最小化 (1 - 相似度)
    loss_intra = torch.sum(loss_intra)

    # 计算与所有原型的相似度
    similarity_all = torch.matmul(rep_norm, protos_norm.t())        # 形状: (batch_size, num_classes)
    similarity_all = torch.abs(similarity_all)

    # 创建掩码，将正确类别的相似度置零
    mask = torch.ones_like(similarity_all)
    mask.scatter_(1, labels.view(-1, 1), 0)                        # 正确类别相似度置零

    # 计算类间损失
    similarity_other = similarity_all * mask                        # 仅保留其他类别的相似度
    loss_inter = torch.sum(similarity_other)                             # 对所有类别求和

    # 合并所有损失
    loss = loss_intra +  loss_inter

    return loss           

def cos_sim(rep, protos):
    """
    参数：
        rep (Tensor): 表征张量，形状为 (batch_size, feature_dim)。
        protos (Tensor): 原型张量，形状为 (num_classes, feature_dim)。
    返回：
        Tensor: 计算得到余弦相似度
    """
    # 归一化表征和原型
    rep_norm = F.normalize(rep, p=2, dim=1)          # 形状: (batch_size, feature_dim)

    protos_norm = F.normalize(protos, p=2, dim=1)   # 形状: (num_classes, feature_dim)

    # 计算与所有原型的相似度
    similarity = torch.matmul(rep_norm, protos_norm.t())        # 形状: (batch_size, num_classes)

    return similarity

class Trainable_prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim), 
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)

        return out