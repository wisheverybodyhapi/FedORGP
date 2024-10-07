import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clienttgp import clientTGP
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

class FedTGP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientTGP)

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
        
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            self.PROTO = Trainable_prototypes(
                self.num_classes, 
                self.server_hidden_dim, 
                self.feature_dim, 
                self.device
            ).to(self.device)
            self.logger.write(self.PROTO)
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

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
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))
        # print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_models()
        
    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

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

        # calculate class-wise minimum distance
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        avg_protos = proto_cluster(uploaded_protos_per_client)
        for k1 in avg_protos.keys():
            for k2 in avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)
        self.min_gap = torch.min(self.gap)
        for i in range(len(self.gap)):
            if self.gap[i] > torch.tensor(1e8, device=self.device):
                self.gap[i] = self.min_gap
        self.max_gap = torch.max(self.gap)
        self.logger.write('class-wise minimum distance: {}'.format(self.gap))
        self.logger.write('min_gap: {}'.format(self.min_gap))
        self.logger.write('max_gap: {}'.format(self.max_gap))

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

    def update_Gen(self):
        Gen_opt = torch.optim.SGD(self.PROTO.parameters(), lr=self.server_learning_rate)
        self.PROTO.train()
        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, self.batch_size, 
                                      drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                y = torch.Tensor(y).type(torch.int64).to(self.device)

                proto_gen = self.PROTO(list(range(self.num_classes)))

                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)
                
                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                gap2 = min(self.max_gap.item(), self.margin_threthold)
                dist = dist + one_hot * gap2
                loss = self.CEloss(-dist, y)

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