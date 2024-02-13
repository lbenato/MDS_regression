import torch
import torch_geometric
from torch_geometric.data import Data
from torch_cluster import knn_graph
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import tensorflow as tf
import pandas as pd

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, graph_inputs):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(9, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels + graph_inputs, 50)
        self.lin2 = Linear(50, 30)
        self.lin3 = Linear(30, 20)
        self.lin4 = Linear(20, 5)
        self.lin5 = Linear(5, 1)

        self.graph_inputs = graph_inputs

    def forward(self, x, edge_index, batch, u):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final sequential layer
        x = F.dropout(x, p=0.2, training=self.training)
        
        """
        print('this is x ', x)
        print('this is u ', u)
        print('size x ', x.size())
        print('size u ', u.size())
        print('this is batch -1 ', batch[-1].item())
        """
        
        # reshaping u
        u = torch.reshape(u, (batch[-1].item() + 1, self.graph_inputs))
        
        # merge pooled node info with graph level info
        x = torch.cat((x,u), dim = 1)
        
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        x = x.relu()
        x = self.lin4(x)
        x = x.relu()
        x = self.lin5(x)
        x = x.relu()
        
        return x


class graph_maker:
    def __init__(self, index = 1):
        self.index = index
        
    
class gcn_model:
    def __init__(self, index = 1):
        self.index = index
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    def test1(self, ext):
        out = self.index + ext
        print(out)
        return out
    
    def make_model(self, hidden_channels = 16, graph_features = 28, learn = 0.01):
        self.model = GCN(hidden_channels = hidden_channels, graph_inputs = graph_features)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learn)
        self.criterion = torch.nn.MSELoss()
        print(self.model)
        
    
    def train(self, train_loader):
        self.model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            self.optimizer.zero_grad()  # Clear gradients.
            out = self.model(data.x, data.edge_index, data.batch, data.u)  # Perform a single forward pass.
            loss = self.criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            

    def test(self, loader):
        self.model.eval()
        mloss = 0
        runs = 0
        for data in loader:  # Iterate in batches over the training/test dataset
            out = self.model(data.x, data.edge_index, data.batch, data.u)
            mloss += self.criterion(out, data.y)
            runs += 1
        mloss /= runs
        return mloss

    def predict_plot(self, predictions, labels, title = None, axlabels = ('clusters', 'generated particle energy')):
        fig, ax = plt.subplots(figsize = (6, 4))

        histdata, bins, dummy = ax.hist(labels, bins = 50, histtype="step", color = 'b', label = 'truth')
        ax.hist(predictions, bins = bins, histtype="step", color = 'r', label = 'predictions')
        ax.set_yscale('log')
        ax.set_ylabel(axlabels[0])
        ax.set_xlabel(axlabels[1])
        ax.legend()
        if title:
            ax.set_title(title)

    def predict(self, loader):
        self.model.eval()
        labels = []
        pred = []
        for data in loader:
            pred.append(self.model(data.x, data.edge_index, data.batch, data.u).item())
            labels.append(data.y.item()) 
            #rint(model(data.x, data.edge_index, data.batch).item())
            #rint(data.y.item())
        self.predict_plot(pred, labels)
        
    def plot_loss(self, train, val):
        fig, ax = plt.subplots(figsize = (6, 4))
        epoch = np.arange(1, len(train) + 1)
        plt.plot(epoch, train, label = 'Training', color = 'blue')
        plt.plot(epoch, val, label = 'Validation',linestyle = '--', color = 'blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.show()

        
class FCN(torch.nn.Module):
    def __init__(self, inputs = 28, layers = [50, 30, 20, 5, 1]):
        super(FCN, self).__init__()
        
        self.fc_layers = torch.nn.ModuleList()  # ModuleList to hold dynamically created layers
        
        # first layer
        self.fc_layers.append(torch.nn.Linear(inputs, layers[0]))
        self.fc_layers.append(torch.nn.ReLU())
        
        # other layers
        for i in range(len(layers) - 1):
            self.fc_layers.append(torch.nn.Linear(layers[i], layers[i + 1]))
            self.fc_layers.append(torch.nn.ReLU())
            
        self.inputs = inputs
        
    def forward(self, x, batch):
        # reshaping x
        x = torch.reshape(x, (batch[-1].item() + 1, self.inputs))
        #print(x.size())
        
        for layer in self.fc_layers:
            x = layer(x)
        return x
    
    
class fcn_model:
    def __init__(self, index = 1):
        self.index = index
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    def test1(self, ext):
        out = self.index + ext
        print(out)
        return out
    
    def make_model(self, inputs = 28, learn = 0.01):
        self.model = FCN(inputs = inputs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learn)
        self.criterion = torch.nn.MSELoss()
        print(self.model)
        
    
    def train(self, train_loader):
        self.model.train()
        self.dummy = True
        for data in train_loader:  # Iterate in batches over the training dataset.
            out = torch.flatten(self.model(data.u, data.batch))  # Perform a single forward pass.
            loss = self.criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            if self.dummy:
                print(data.u)
                print(out, data.y)
                self.dummy = False
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
            
    def test(self, loader):
        self.model.eval()
        mloss = 0
        runs = 0
        for data in loader:  # Iterate in batches over the training/test dataset
            out = self.model(data.u, data.batch)
            mloss += self.criterion(out, data.y)
            runs += 1
        mloss /= runs
        return mloss
    
    def predict_plot(self, predictions, labels, title = None, axlabels = ('clusters', 'generated particle energy')):
        fig, ax = plt.subplots(figsize = (6, 4))

        histdata, bins, dummy = ax.hist(labels, bins = 50, histtype="step", color = 'b', label = 'truth')
        ax.hist(predictions, bins = bins, histtype="step", color = 'r', label = 'predictions')
        ax.set_yscale('log')
        ax.set_ylabel(axlabels[0])
        ax.set_xlabel(axlabels[1])
        ax.legend()
        if title:
            ax.set_title(title)

    def predict(self, loader):
        self.model.eval()
        labels = []
        pred = []
        for data in loader:
            pred.append(self.model(data.x, data.batch).item())
            labels.append(data.y.item()) 
            #rint(model(data.x, data.edge_index, data.batch).item())
            #rint(data.y.item())
        self.predict_plot(pred, labels)
        
    def plot_loss(self, train, val):
        fig, ax = plt.subplots(figsize = (6, 4))
        epoch = np.arange(1, len(train) + 1)
        plt.plot(epoch, train, label = 'Training', color = 'blue')
        plt.plot(epoch, val, label = 'Validation',linestyle = '--', color = 'blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.show()