"""
Graph Neural Network for Chess 
Input : Board state
Output : Value of the board state and probability of each move
"""
import os
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F, Linear, BatchNorm1d, ModuleList, ReLU, Sequential
from torch_geometric.nn.glob import GlobalAttention
from torch_geometric.data import Data
from utils.GAT import GAT
import torch

class GNN(pl.LightningModule):
    def __init__(self, config, data_dir=None,name='GNN'):
        super(GNN, self).__init__()
        self.name = name
        self.data_dir = data_dir or os.getcwd()  # pass this from now on
        self.learning_rate = config['lr']
        self.hidden_size = config['hidden']
        self.num_layers = config['n_layers']

        self.gnn = GAT(20, self.hidden_size, num_layers=self.num_layers, edge_dim=2,v2=True,heads=8, norm=nn.BatchNorm1d(self.hidden_size),act_first=True)

        self.pool = GlobalAttention(gate_nn=nn.Linear(self.hidden_size, 1))

        self.fc1 = Linear(self.hidden_size, self.hidden_size)
        # self.fc2 = Linear(self.hidden_size, self.hidden_size)

        # self.fc3 = Linear(self.hidden_size, self.hidden_size)
        # self.fc4 = Linear(self.hidden_size, self.hidden_size)
        self.bn = nn.BatchNorm1d(self.hidden_size)
        # self.fc5 = Linear(self.hidden_size, self.hidden_size)
        # self.fc6 = Linear(self.hidden_size, self.hidden_size)
        # self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.policy_out = Linear(self.hidden_size, self.hidden_size)
        self.fc7 = Linear(self.hidden_size, int(self.hidden_size/2))
        self.fc8 = Linear(int(self.hidden_size/2), int(self.hidden_size/6))
        self.value_out = Linear(int(self.hidden_size/6), 1)
        self.emb_f = None
        self.save_hyperparameters()


    def forward(self, graphs: Data):
        graphs = graphs[0]
        x = graphs.x.to(torch.float)
        edge_attr = graphs.edge_attr.to(torch.float)
        edge_index = graphs.edge_index

        x = F.relu(self.gnn(x, edge_index, edge_attr))
        self.emb_f = self.pool(x, graphs.batch)
        x = F.relu(self.fc1(self.emb_f))
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = F.relu(self.fc4(x))
        x = self.bn(x)
        # x = self.fc5(x)
        # x = F.relu(self.fc6(x))
        # x = self.bn1(x)
        x = F.log_softmax(self.policy_out(x),dim=1).exp()
        policy = torch.reshape(x,(x.shape[0],8,8,73))
        x = self.fc7(x)
        # x = BatchNorm1d(x.shape[1])(x)
        x = F.relu(self.fc8(x))
        
        value = torch.tanh(self.value_out(x))
        return value, policy

    def mse_loss(self, prediction, target):
        prediction = prediction.reshape(target.shape)
        result = F.mse_loss(prediction, target)
        return result

    def cross_entropy(self, prediction, target):
        # prediction = prediction.reshape(target.shape)
        result = F.cross_entropy(prediction, target)
        return result

    def training_step(self, train_batch, batch_idx):
        value, policy = self.forward(train_batch)
        value_loss = self.mse_loss(value, train_batch[1])
        policy_loss = self.cross_entropy(policy, train_batch[2])
        loss = value_loss + policy_loss
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_bottleneck(self):
        return self.emb_f



##### Testing #####
# from torch_geometric.data import Dataset
# from torch_geometric.loader import DataLoader
# from datamodule import ChessDataset
# config = {'lr': 0.001, 'hidden': 4672, 'n_layers': 16, 'batch_size': 2}
# temp = ChessDataset(fens=['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1','rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],values=torch.tensor([0,0]).to(torch.float),policies=torch.tensor([np.random.uniform(size=(8,8,73)),np.random.uniform(size=(8,8,73))]).to(torch.float))
# temp_dl = DataLoader(temp, batch_size=config['batch_size'], shuffle=True, num_workers=0)
# model = GNN(config)
# trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=10)
# trainer.fit(model, temp_dl)