import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client, load_item, save_item
from flcore.trainmodel.models import BaseHeadSplit


class clientFML(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.alpha = args.alpha
        self.beta = args.beta

        self.KL = nn.KLDivLoss()
        self.global_model = BaseHeadSplit(args, 0).to(args.device)

    def train(self):
        trainloader = self.load_train_data()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        optimizer_g = torch.optim.SGD(self.global_model.parameters(), lr=self.learning_rate)
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                output_g = self.global_model(x)
                loss = self.loss(output, y) * self.alpha + self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) * (1-self.alpha)
                loss_g = self.loss(output_g, y) * self.beta + self.KL(F.log_softmax(output_g, dim=1), F.softmax(output, dim=1)) * (1-self.beta)

                optimizer.zero_grad()
                optimizer_g.zero_grad()
                loss.backward(retain_graph=True)
                loss_g.backward()
                # prevent divergency on specifical tasks
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 10)
                optimizer.step()
                optimizer_g.step()


        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, global_model):
        self.global_model = copy.deepcopy(global_model)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
        
        return test_acc, test_num, 0

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
                output = self.model(x)
                output_g = self.global_model(x)
                loss = self.loss(output, y) * self.alpha + self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) * (1-self.alpha)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num