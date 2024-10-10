import copy
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientgen import clientGen
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.model_utils import check_for_model
from threading import Thread


class FedGen(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGen)

        self.logger.write(f"Join ratio / total clients: {self.join_ratio} / {self.num_clients}")
        self.logger.write("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        if self.save_folder_name == 'temp' or check_for_model(self.model_folder_name) == False:
            self.generative_model = Generative(
                                    args.noise_dim, 
                                    args.num_classes, 
                                    args.hidden_dim, 
                                    args.feature_dim, 
                                    self.device
                                ).to(self.device)
        else:
            self.generative_model = load_item(self.role, 'generative_model', self.model_folder_name)

        self.loss = nn.CrossEntropyLoss()
        self.generator_learning_rate = args.generator_learning_rate
        
        self.qualified_labels = []
        for client in self.clients:
            for yy in range(self.num_classes):
                self.qualified_labels.extend([yy for _ in range(int(client.sample_per_class[yy].item()))])
        for client in self.clients:
            client.qualified_labels = self.qualified_labels

        self.server_epochs = args.server_epochs

        self.head = copy.deepcopy(self.clients[0].model.head)

        

    def train(self):
        for i in range(self.start_round, self.end_round):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_heads()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_ids()
            self.train_generator()
            self.aggregate_parameters()

            if i%self.eval_gap == 0:
                self.logger.write(f"-------------Round number: {i}-------------")
                self.logger.write("Evaluate heterogeneous models")
                self.evaluate()

            self.Budget.append(time.time() - s_t)
            self.logger.write("The current global round takes {} seconds".format(self.Budget[-1]))

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        self.logger.write("Best accuracy: {}".format(max(self.rs_test_acc)))
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))
        # print("Average time cost per round.")
        # print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_models()

    def send_heads(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_head(self.head)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.head)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def train_generator(self):
        generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=self.generator_learning_rate, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.generative_model.train()

        for _ in range(self.server_epochs):
            labels = np.random.choice(self.qualified_labels, self.batch_size)
            labels = torch.LongTensor(labels).to(self.device)
            z = self.generative_model(labels)

            logits = 0
            for w, model in zip(self.uploaded_weights, self.uploaded_models):
                model.eval()
                logits += model(z) * w

            generative_optimizer.zero_grad()
            loss = self.loss(logits, labels)
            loss.backward()
            generative_optimizer.step()

    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        client = self.clients[self.uploaded_ids[0]]
        for param in self.head.parameters():
            param.data.zero_()
            
        for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            client_model = client.model.head
            for server_param, client_param in zip(self.head.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

        

# based on official code https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/trainmodel/generator.py
class Generative(nn.Module):
    def __init__(self, noise_dim, num_classes, hidden_dim, feature_dim, device) -> None:
        super().__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device

        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU()
        )

        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, labels):
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim), device=self.device) # sampling from Gaussian

        y_input = F.one_hot(labels, self.num_classes)
        z = torch.cat((eps, y_input), dim=1)

        z = self.fc1(z)
        z = self.fc(z)

        return z