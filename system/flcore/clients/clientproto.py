import copy
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict


class clientProto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
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

        self.protos = defaultdict(list)
        # 假设 max_local_epochs 是最大训练轮数
        for step in range(max_local_epochs):  # tqdm 包裹外层循环
            for i, (x, y) in enumerate(trainloader):  # tqdm 包裹内层数据加载
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))  # 模拟慢训练
                
                rep = self.model.base(x)  # 获取模型的基础输出
                output = self.model.head(rep)  # 获取模型的最终输出
                loss = self.loss(output, y)  # 计算损失
                
                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda  # 添加均方误差损失
                
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    self.protos[y_c].append(rep[i, :].detach().data)  # 存储原型
                
                optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重

        self.agg_func()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def get_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        self.protos = defaultdict(list)
        for i, (x, y) in enumerate(trainloader):  # tqdm 包裹内层数据加载
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            
            rep = self.model.base(x)  # 获取模型的基础输出
            
            for i, yy in enumerate(y):
                y_c = yy.item()
                self.protos[y_c].append(rep[i, :].detach().data)  # 存储原型

        self.agg_func()

    def set_protos(self, global_protos):
        # receive global protos
        self.global_protos = global_protos

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

                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in self.global_protos.items():
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r, pro)

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
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
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


    # https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
    def agg_func(self):
        for [label, proto_list] in self.protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                self.protos[label] = proto / len(proto_list)
            else:
                self.protos[label] = proto_list[0]