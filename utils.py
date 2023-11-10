import datetime as dt
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx

def make_dir(path):
    path_list = path.split('/')
    now_path = ""
    for i, dir in enumerate(path_list):
       if i == 0:
          continue
       else:
          now_path += f"/{dir}"
          if not os.path.exists(now_path):
             os.makedirs(now_path)


class EarlyStopping():
    def __init__(self, config):
        self.config = config
        model_path = os.path.join(os.getcwd(), 'log', 'models')
        make_dir(model_path)
        now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.file_nm = os.path.join(model_path, f'{now}_model.pt')

        self.best_loss = float('inf')
        self.best_model = None
        self.patience = 0
    
    def save_model(self):
        torch.save(self.best_model.state_dict, self.file_nm)
    
    def check(self, loss, model, epoch):
        bool = False

        if loss < self.best_loss:
            self.best_loss = loss
            self.patience = 0
            self.best_model = model
        
        else:
            self.patience += 1
            if self.patience >= self.config['train']['patience']:
                print("Early Stopping.", flush=True)
                print(f"Best valid loss: {self.best_loss:.4f}", flush=True)
                self.save_model()
                bool = True
        
        if epoch + 1 == self.config['train']['epochs']:
            self.save_model()
        
        return bool

class Visualize():
    def __init__(self):
        self.now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.parent_path = os.path.join(os.getcwd(), 'log/img')
        make_dir(self.parent_path)
    
    def save_loss(self, train_loss, valid_loss):
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='train')
        plt.plot(valid_loss, label='valid')
        plt.title("losses")
        plt.ylabel('loss')
        plt.xlabel("epoch")
        plt.legend()

        file_nm = os.path.join(self.parent_path, f"{self.now}_loss.png")
        plt.savefig(file_nm)
        plt.close()
    
    def save_result(self, model, dataset, color_map="Set3"):
        model.eval()
        _, pred = model(dataset.data).max(dim=1)
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 16))
        axes = axes.flatten()
        G = to_networkx(dataset.data, to_undirected=True)
        color_map = plt.cm.tab10(dataset.data.y / dataset.num_classes)
        pos = nx.spring_layout(G, seed=25)
        nx.draw(
            G, 
            pos, 
            ax=axes[0],
            node_color=color_map, 
            with_labels=False,
            node_size=20,
            font_weight='bold'
        )
        axes[0].axis('off')

        G = to_networkx(dataset.data, to_undirected=True)
        color_map = plt.cm.tab10(pred / dataset.num_classes)
        pos = nx.spring_layout(G, seed=25)
        nx.draw(
            G, 
            pos, 
            ax=axes[1],
            node_color=color_map, 
            with_labels=False,
            node_size=20,
            font_weight='bold'
        )
        axes[1].axis('off')

        file_nm = os.path.join(self.parent_path, f"{self.now}_result.png")
        plt.tight_layout()
        plt.savefig(file_nm)
        plt.close()