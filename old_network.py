"""
Graph Neural Network for Chess 
Input : Board state
Output : Value of the board state and probability of each move
"""
import os
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F, Linear, BatchNorm1d, ModuleList, ReLU, Sequential
# from torch_geometric.nn.glob import GlobalAttention
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.data import Data
from utils.GAT import GAT, GraphSAGE
# from torch_geometric.nn.models import EdgeCNN
from torch_geometric.nn.models import PNA
import torch
import numpy as np
from torchvision.models import resnet50
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Sequential(
                        nn.Linear(in_channels, out_channels),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU())
        self.fc2 = nn.Sequential(
                        nn.Linear(out_channels, out_channels),
                        nn.BatchNorm1d(out_channels))
        self.fc3 = nn.Sequential(
                        nn.Linear(out_channels, out_channels),
                        nn.BatchNorm1d(out_channels))
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        out = self.fc1(x)
        residual = out
        out = self.fc2(out)
        out += residual
        out = self.fc3(out)
        out = self.relu(out)
        return out
    
class ResidualNet(nn.Module):
    def __init__(self,in_dim,out_dim,hidden_dim,n_layers) -> None:
        super(ResidualNet, self).__init__()

        self.in_layer = ResidualBlock(in_dim,hidden_dim)
        self.hidden_layers = nn.ModuleList([ResidualBlock(hidden_dim,hidden_dim) for _ in range(n_layers)])
        self.out_layer = nn.Linear(hidden_dim,out_dim)
    def forward(self,x):
        out = self.in_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.out_layer(out)
        return out
class Network(pl.LightningModule):
    def __init__(self, config, data_dir=None,name='GNN'):
        super(Network, self).__init__()
        self.name = name
        self.data_dir = data_dir or os.getcwd()  # pass this from now on
        self.learning_rate = config['lr']
        self.hidden_size = config['hidden']
        self.num_layers = config['n_layers']
        self.board_rep = config['board_representation']
        self.gnn_type = config['gnn_type']
        self.use_resnet = config['useresnet']
        if self.board_rep == 'graph':
            self.gnn = GAT(20, 1024, num_layers=self.num_layers, edge_dim=2,v2=True,heads=config['heads'], norm=nn.BatchNorm1d(1024),act_first=True)
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(1024, 1))
            self.resnet = ResidualNet(1024,self.hidden_size,2048,6)
            self.val_resnet = ResidualNet(1024,1,50,5) 

        elif self.board_rep == 'array': 
            if self.use_resnet:
                self.conv1 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)
                self.resnet = resnet50(weights=None)
                self.fc = nn.Linear(1000, self.hidden_size)
            else:
                self.pre_fc1 = nn.Linear(np.prod([8,8,21]), self.hidden_size)
                self.pre_bn = nn.BatchNorm1d(self.hidden_size)
                self.pre_fc2 = nn.Linear(self.hidden_size, self.hidden_size)
                self.pre_fc3 = nn.Linear(self.hidden_size, self.hidden_size)

            self.dropout = nn.Dropout(0.2)
            self.fc1 = Linear(self.hidden_size, self.hidden_size)
            # self.fc2 = Linear(self.hidden_size, self.hidden_size)
            # self.fc3 = Linear(self.hidden_size, self.hidden_size)
            self.fc4 = Linear(self.hidden_size, self.hidden_size)
            # self.bn = nn.BatchNorm1d(self.hidden_size)
            self.bn1 = nn.BatchNorm1d(self.hidden_size)
            # self.fc6 = Linear(self.hidden_size, self.hidden_size)
            self.bn2 = nn.BatchNorm1d(int(self.hidden_size/6))
            self.last = Linear(self.hidden_size, self.hidden_size)
            size = 1000 if self.board_rep == 'array' and self.use_resnet else self.hidden_size
            self.piece_pred = Linear(self.hidden_size, 12)
            self.sq_pred =  Linear(self.hidden_size, 64)
            self.policy_out = Linear(size+76, self.hidden_size)
            self.fc7 = Linear(size, int(self.hidden_size/2))
            self.fc8 = Linear(int(self.hidden_size/2), int(self.hidden_size/6))
            self.fc9 = Linear(int(self.hidden_size/6), int(self.hidden_size/10))
            self.value_out = Linear(int(self.hidden_size/10), 1)
        self.save_hyperparameters()


    def forward(self, data):
        if self.board_rep == 'graph':
            graphs = data[0]
            x = graphs.x.to(torch.float)
            edge_attr = graphs.edge_attr.to(torch.float)
            edge_index = graphs.edge_index
            emb = F.relu(self.gnn(x=x, edge_index=edge_index, edge_attr=edge_attr))
            # graph_edge_indices = self.get_edge_indices_for_all_graphs(edge_index,graphs.batch)
            # policy = F.softmax(self.resnet(emb),dim=1)
            policy = self.resnet(emb)
            policy = policy.view((policy.shape[0],8,8,73))
            value = torch.tanh(self.val_resnet(emb))
            return value,policy
        elif self.board_rep == 'array':
            arrays = data[0]
            arrays = arrays if type(arrays) != np.ndarray else torch.tensor(arrays)
            arrays = arrays if len(arrays.shape) != 3 else arrays.unsqueeze(0)
            x = arrays.to(torch.float)
            if self.use_resnet:
                x = self.conv1(x)
                emb = self.resnet(x)
                x = self.fc(emb)
            else:
                x = self.pre_fc1(x.view(x.shape[0],1344))
                x  = self.pre_bn(x)
                emb = self.pre_fc2(x)
                x = self.pre_fc3(emb)
            piece_pred = F.softmax(self.piece_pred(x),dim=1) # (1,12)
            sq_pred = F.softmax(self.sq_pred(x),dim=1).view((piece_pred.shape[0],8,8)) # (1,8,8)
            conc = torch.concat([piece_pred,sq_pred.reshape(sq_pred.shape[0],64),emb],dim=1)
            x = self.policy_out(conc)
            policy = x.view((x.shape[0],8,8,73))
            # res = x
            # x = F.relu(self.fc2(x))
            # x = self.bn(x)
            # x = F.relu(self.fc3(x))
            # x = x + res
            x = self.fc7(emb)
            x = F.relu(self.fc8(x))
            x = self.bn2(x)
            x = F.relu(self.fc9(x))
            value = torch.tanh(self.value_out(x))
            return value, policy,piece_pred,sq_pred

    def mse_loss(self, prediction, target):
        # prediction = prediction.view(target.shape)
        result = F.mse_loss(prediction, target)
        return result

    def cross_entropy(self, prediction, target):
        # prediction = prediction.reshape(target.shape)
        prediction = prediction.view(prediction.shape[0],torch.prod(torch.tensor(prediction.shape)[1:]))
        target = target.view(target.shape[0], torch.prod(torch.tensor(target.shape)[1:]))
        result = F.cross_entropy(prediction,target)
        return result

    def training_step(self, train_batch, batch_idx):
        value, policy,piece_pred,sq_pred = self.forward(train_batch)
        value_loss = self.mse_loss(value, train_batch[1].view(value.shape))
        policy_loss = self.cross_entropy(policy, train_batch[2]) 
        piece_loss = self.cross_entropy(piece_pred, train_batch[3]) 
        sq_loss = self.cross_entropy(sq_pred, train_batch[4]) 
        loss = policy_loss + value_loss + piece_loss + sq_loss
        # print('real value: ', train_batch[1].view(value.shape))
        # print('predicted value: ', value)
        print('Real, Pred: ')
        print([(float(train_batch[1][i]),float(value[i])) for i in range(value.shape[0])])
        print('loss: ', float(loss))
        print('mse loss: ', float(value_loss))
        print('cross entropy loss: ', float(policy_loss), float(piece_loss),float(sq_loss))

        self.log('train_loss', loss)
        return loss

    def test_step(self, test_batch,batch_idx):
        value, policy,_,_= self.forward(test_batch)
        value_loss = self.mse_loss(value, test_batch[1].view(value.shape))
        self.log_dict({'mse':value_loss})
        leng = int(test_batch[2].size()[0])
        move_accs = [True if list(np.unravel_index(np.argmax(policy[i]).tolist(),(8,8,73))) in np.nonzero(test_batch[2][i]).tolist() else False for i in range(leng)]
        # acc = sum(move_accs)/len(move_accs)
        acc = sum(move_accs)/len(move_accs)
        self.log_dict({'mse':float(value_loss),'move_acc':acc},batch_size=1)
        return value_loss,acc


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,weight_decay=0.0001)
        return optimizer

if __name__ == '__main__':

    ##### Testing #####
    # from torch_geometric.data import Dataset
    # from torch_geometric.loader import DataLoader
    from torch.utils.data import DataLoader
    from datamodule import ChessDataset, ChessDataset_arrays
    import chess
    config = {'lr': 0.0003 , 'hidden': 4672, 'n_layers': 1,'heads': 64,'gnn_type':None,'board_representation':'array','useresnet':False}
    temp = ChessDataset_arrays(boards=[chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'),chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')],values=torch.tensor([0,0]).to(torch.float),policies=torch.tensor([np.random.uniform(size=(8,8,73)),np.random.uniform(size=(8,8,73))]).to(torch.float))
    temp_dl = DataLoader(temp, batch_size=2, shuffle=True, num_workers=0)
    model = Network(config)
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=10)
    trainer.fit(model, temp_dl)