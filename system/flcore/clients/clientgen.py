import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item


class clientGen(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        trainloader = self.load_train_data()
        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1

        self.qualified_labels = []
        self.generative_model = None
        

    def train(self):
        trainloader = self.load_train_data()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
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
                loss = self.loss(output, y)
                
                if self.generative_model is not None:
                    labels = np.random.choice(self.qualified_labels, self.batch_size)
                    labels = torch.LongTensor(labels).to(self.device)
                    z = self.generative_model(labels)
                    loss += self.loss(self.model.head(z), labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
            
        
    def set_head(self, head):
        for new_param, old_param in zip(head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()

    def train_metrics(self):
        trainloader = self.load_train_data()
        # model.to(self.device)
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
                loss = self.loss(output, y)
                
                if self.generative_model is not None:
                    labels = np.random.choice(self.qualified_labels, self.batch_size)
                    labels = torch.LongTensor(labels).to(self.device)
                    z = self.generative_model(labels)
                    loss += self.loss(self.model.head(z), labels)
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
