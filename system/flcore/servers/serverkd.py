import copy
import random
import time
import os
import numpy as np
from flcore.clients.clientkd import clientKD, recover, decomposition
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from flcore.trainmodel.models import BaseHeadSplit
from utils.model_utils import check_for_model


class FedKD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        if self.save_folder_name == 'temp' or check_for_model(self.model_folder_name) == False:
            self.global_model = BaseHeadSplit(args, 0).to(args.device)      
        else:
            self.global_model = load_item(self.role, 'global_model', self.model_folder_name)      
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientKD)

        self.logger.write(f"Join ratio / total clients: {self.join_ratio} / {self.num_clients}")
        self.logger.write("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.T_start = args.T_start
        self.T_end = args.T_end
        self.energy = self.T_start
        self.compressed_param = None


    def train(self):
        for i in range(self.start_round, self.end_round):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_ids()
            self.aggregate_parameters()
            self.send_parameters()

            if i%self.eval_gap == 0:
                self.logger.write(f"-------------Round number: {i}-------------")
                self.logger.write("Evaluate heterogeneous models")
                self.evaluate()

            self.Budget.append(time.time() - s_t)
            self.logger.write("The current global round takes {} seconds".format(self.Budget[-1]))

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            self.energy = self.T_start + ((1 + i) / self.global_rounds) * (self.T_end - self.T_start)
            for client in self.clients:
                client.energy = self.energy

        self.logger.write("Best accuracy: {}".format(max(self.rs_test_acc)))
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))
        # print("Average time cost per round.")
        # print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_models()

    def save_models(self):
        # save models
        if self.save_folder_name != 'temp':
            if os.path.exists(self.model_folder_name) == False:
                os.makedirs(self.model_folder_name)
            try:
                for client in self.clients:
                    save_item(client.model, client.role, 'model', self.model_folder_name)
                    save_item(client.global_model, client.role, 'global_model', self.model_folder_name)
                    save_item(client.W_h, client.role, 'W_h', self.model_folder_name)
                self.logger.write('finish saving models/W_h/global_models of clients')
                save_item(self.global_model, self.role, 'global_model', self.model_folder_name)
                self.logger.write('finish saving global_model of server')
            except Exception as e:
                self.logger.write(f"An error occurred: {str(e)}")
                self.logger.logger.exception("Exception occurred while saving models and global_model")

    def send_parameters(self):
        for client in self.clients:
            client.set_parameters(self.compressed_param)

    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        global_param = {name: param.detach().cpu().numpy() 
                        for name, param in self.global_model.named_parameters()}
        for k in global_param.keys():
            global_param[k] = np.zeros_like(global_param[k])
            
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            compressed_param = client.compressed_param
            client_param = recover(compressed_param)
            for server_k, client_k in zip(global_param.keys(), client_param.keys()):
                global_param[server_k] += client_param[client_k] * 1/len(self.uploaded_ids)

        self.compressed_param = decomposition(global_param.items(), self.energy)