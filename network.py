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
from torch_geometric.nn.pool.edge_pool import EdgePooling
from utils.batch_handling import get_edge_indices_for_all_graphs, split_by_graph, softmax_each_group, combine_edge_and_graph_embeddings
from array_net import ChessNet, AlphaLoss
from array_net import move_acc as array_move_acc

def calculate_dimensions(in_dim, out_dim, hidden_dim,n_layers):
    dimensions = [in_dim]

    # Start halving the input dimension until we reach the hidden dimension.
    while in_dim > hidden_dim:
        in_dim = in_dim // 2
        dimensions.append(in_dim)

    # Then append a certain number of hidden layers..
    dimensions.extend([hidden_dim] * n_layers)

    # Finally, we gradually decrease the dimension to the output dimension.
    while hidden_dim > out_dim:
        hidden_dim = hidden_dim // 2
        if hidden_dim!=out_dim:
            dimensions.append(hidden_dim)

    dimensions.append(out_dim)
    return dimensions
class PPOClipLoss(nn.Module):
    def __init__(self, epsilon=0.2, value_loss_coef=1.0, entropy_coef=0.01):
        super(PPOClipLoss, self).__init__()
        self.epsilon = epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
    # install torchrl
    def forward(self, true_value, pred_value, true_policy, pred_policy):
        # Get corresponding true policy slices
        valid_indices = (true_policy != -1)
        true_policy = true_policy[valid_indices].view(-1)
        # Calculate the starting indices of each batch
        batch_sizes = [pred.shape[0] for pred in pred_policy]
        batch_start_indices = [0] + batch_sizes[:-1]
        batch_start_indices = torch.cumsum(torch.tensor(batch_start_indices), dim=0)
        true_policy_slices = [true_policy[start_idx:start_idx + size] for start_idx, size in zip(batch_start_indices, batch_sizes)]
        # Compute KL divergence
        # kl_divs = [F.kl_div(F.log_softmax(pred.view(-1),dim=0), F.log_softmax(true_slice.view(-1),dim=0),log_target=True) for pred, true_slice in zip(pred_policy, true_policy_slices)]
        kl_divs = [F.cross_entropy(pred.view(-1),true_slice.view(-1)) for pred, true_slice in zip(pred_policy, true_policy_slices)]
        policy_loss = torch.stack(kl_divs).mean()
        value_loss = F.mse_loss(pred_value.view(-1), true_value.view(-1))
        loss = policy_loss + self.value_loss_coef * value_loss
        print('Value loss: ', value_loss.item())
        print('Policy loss: ', policy_loss.item())
        return loss
class FCBlock(nn.Module):
    def __init__(self, size):
        super(FCBlock, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.bn1 = nn.BatchNorm1d(size)

    def forward(self, s):
        s = s.view(s.size(0), -1)  # Flatten the tensor
        s = s.float()
        s = F.relu(self.bn1(self.fc1(s)))
        return s

class ResBlockFCN(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlockFCN, self).__init__()
        self.fc1 = nn.Linear(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.fc2 = nn.Linear(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = F.relu(self.bn1(out))
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class ResNetFCN(nn.Module):
    def __init__(self, in_size, out_size, n_layers, head_type='Policy',softmax=True) -> None:
        super(ResNetFCN, self).__init__()
        self.n_layers = n_layers
        self.pol_fc = FCBlock(size=in_size)
        self.head_type = head_type
        for block in range(n_layers):
            setattr(self, "res_%i" % block, ResBlockFCN(inplanes=in_size, planes=in_size))
        self.fc1 = nn.Linear(in_size, in_size // 2)
        self.bn1 = nn.BatchNorm1d(in_size // 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc2 = nn.Linear(in_size // 2, out_size)
        if head_type == 'Value':
            self.fc3 = nn.Linear(out_size, out_size)
            self.relu = nn.ReLU()
        self.softmax = softmax

    def forward(self, x):
        x = self.pol_fc(x)
        for block in range(self.n_layers):
            x = getattr(self, "res_%i" % block)(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        if self.head_type == 'Value':
            x = self.fc3(x)
            x = self.relu(x)
        elif self.head_type == 'Policy' and self.softmax:
            x = self.logsoftmax(x).exp()
        return x
        
class GlobalAttention(nn.Module):
    def __init__(self, graph_emb_size, edge_emb_size, num_heads,emb_size):
        super(GlobalAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads)
        self.linear_in = nn.Linear(graph_emb_size + edge_emb_size, emb_size)
        self.linear_out = nn.Linear(emb_size, edge_emb_size)

    def forward(self, graph_emb, edge_emb, graph_edge_indices):
        graph_emb_expanded = graph_emb[graph_edge_indices]  # (n_edges, graph_emb_size)
        # Concatenate graph and edge embeddings along the feature dimension
        concat_emb = torch.cat([graph_emb_expanded, edge_emb], dim=-1)  # (n_edges, graph_emb_size + edge_emb_size)

        # Project concatenated embeddings to the input size expected by the attention mechanism
        projected_emb = self.linear_in(concat_emb)

        # Apply attention using projected embeddings as key and value, and edge embeddings as query
        attn_output, _ = self.multihead_attention(
            projected_emb.unsqueeze(0),
            projected_emb.unsqueeze(0),
            projected_emb.unsqueeze(0)
        )
        attn_output = attn_output.squeeze(0)

        # Project attention output back to the original edge embedding size
        attn_output = self.linear_out(attn_output)

        # Normalize the output to obtain a distribution along the edges for each graph independently
        attn_output = softmax_each_group(attn_output, graph_edge_indices)

        return attn_output

class Network(pl.LightningModule):
    def __init__(self, config, data_dir=None,name='GNN'):
        super(Network, self).__init__()
        self.name = name
        self.data_dir = data_dir or os.getcwd()  # pass this from now on
        self.learning_rate = config['lr']
        self.board_rep = config['board_representation']
        if self.board_rep == 'graph':
            self.hiddensize_graph = config['hidden_graph']
            self.hiddensize_edge = config['hidden_edge']
            self.num_layers = config['n_layers']
            self.value_nlayers = config['value_nlayers']
            self.pol_nlayers = config['pol_nlayers']
            self.policy_format = config['policy_format']
            self.GAT_heads_graph = config['heads_GAT_graph']
            self.GAT_heads_edge = config['heads_GAT_edge']
            GAheads = config['GAheads']
            att_emb_size = config['att_emb_size']
            n_edge_f= 3
            graphgnn_size = self.hiddensize_graph
            edgegnn_size = self.hiddensize_edge
            if self.policy_format == 'graph':
                self.edge_gnn = GAT(25, edgegnn_size, num_layers=self.num_layers, edge_dim=n_edge_f,v2=True,heads=self.GAT_heads_edge, norm=nn.BatchNorm1d(edgegnn_size),act_first=True)
                self.graph_gnn = GAT(25, graphgnn_size, num_layers=self.num_layers, edge_dim=n_edge_f,v2=True,heads=self.GAT_heads_graph, norm=nn.BatchNorm1d(graphgnn_size),act_first=True)
                self.pool = AttentionalAggregation(gate_nn=nn.Linear(graphgnn_size, 1))
                # self.edgepool = AttentionalAggregation(gate_nn=nn.Linear(edgegnn_size, 1))
                edge_emb_size = edgegnn_size*2+n_edge_f
                self.global_attention = GlobalAttention(graph_emb_size=graphgnn_size, edge_emb_size=edge_emb_size, num_heads=GAheads,emb_size=att_emb_size)
                self.pol_resnet = ResNetFCN(in_size=edge_emb_size, out_size=1, n_layers=self.pol_nlayers,head_type='Policy',softmax=False)
                self.val_resnet = ResNetFCN(in_size=graphgnn_size, out_size=1, n_layers=self.value_nlayers,head_type='Value')
            elif self.policy_format == 'array':
                graphgnn_size = 512
                self.graph_gnn = GAT(25, graphgnn_size, num_layers=self.num_layers, edge_dim=n_edge_f,v2=True,heads=config['heads'], norm=nn.BatchNorm1d(graphgnn_size),act_first=True)
                self.pool = AttentionalAggregation(gate_nn=nn.Linear(graphgnn_size, 1))
                self.pol_head = ResNetFCN(in_size=graphgnn_size, out_size=8*8*73, n_layers=self.pol_nlayers,head_type='Policy',softmax=True)
                self.val_resnet = ResNetFCN(in_size=graphgnn_size, out_size=1, n_layers=self.value_nlayers,head_type='Value')
                self.alpha_loss = AlphaLoss()

        elif self.board_rep == 'array':
            self.alpha_loss = AlphaLoss()
            self.chess_net = ChessNet()
        self.save_hyperparameters()

    def forward(self, data):
        if self.board_rep == 'array':
            data = data[0]
            data = data.view(-1, 21, 8, 8) 
            policy, value = self.chess_net(data)
            return value, policy
        elif self.board_rep == 'graph':
            if self.policy_format == 'graph':
                graphs = data[0]
                x = graphs.x.to(torch.float)
                edge_attr = graphs.edge_attr.to(torch.float)[:,:3]
                edge_index = graphs.edge_index
                selected_indices = ((edge_attr[:, 0] == 1) & (edge_attr[:, 1] == 0)).nonzero(as_tuple=True)[0]
                edge_emb = self.edge_gnn(x=x, edge_index=edge_index, edge_attr=edge_attr) #(n_nodes, 512)
                graph_emb = self.graph_gnn(x=x, edge_index=edge_index, edge_attr=edge_attr) 
                x_src, x_dst = edge_emb[edge_index[0][selected_indices]], edge_emb[edge_index[1][selected_indices]] #(n_edges, 512), #(n_edges, 512)
                edge_feat = torch.cat([x_src, edge_attr[selected_indices], x_dst], dim=-1) #(n_edges,512 + 3 + 512)
                graph_edge_indices = get_edge_indices_for_all_graphs(edge_index[:,selected_indices],graphs.batch) if graphs.batch is not None else torch.zeros(edge_index[:,selected_indices].shape[1], dtype=torch.long)
                graphs.batch = torch.zeros(graphs.num_nodes, dtype=torch.long) if graphs.batch is None else graphs.batch # if one graph in batch 
                graph_emb = self.pool(graph_emb, graphs.batch)  # (n_graphs, 1024)
                # Apply attention operation to graph and edge embeddings
                attn_output = self.global_attention(graph_emb, edge_feat, graph_edge_indices)  # --> (n_edges, edge_emb_size)
                attn_output = torch.concat(attn_output,dim=0)
                # Apply residual network to attention output to get policy
                policy_logits = self.pol_resnet(attn_output)  # --> (n_edges, 1)
                policy_logits = split_by_graph(policy_logits, graph_edge_indices) 
                value = torch.tanh(self.val_resnet(graph_emb))
            elif self.policy_format == 'array':
                graphs = data[0]
                x = graphs.x.to(torch.float)
                edge_attr = graphs.edge_attr.to(torch.float)[:,:3]
                edge_index = graphs.edge_index
                graph_emb = self.graph_gnn(x=x, edge_index=edge_index, edge_attr=edge_attr) 
                graph_emb = self.pool(graph_emb, graphs.batch)  # (n_graphs, 1024)
                value = torch.tanh(self.val_resnet(graph_emb))
                policy_logits = self.pol_head(graph_emb)
            return value,policy_logits

    def mse_loss(self, prediction, target):
        # prediction = prediction.view(target.shape)
        result = F.mse_loss(prediction.view(-1), target.view(-1))
        return result

    def cross_entropy(self, prediction, target,show=False):
        # beg = 0
        # results = []
        # valid_indices = (target != -1)
        # target = target[valid_indices].view(-1)
        # for i in range(len(prediction)):
        #     res = F.cross_entropy(prediction[i].view(-1),target[beg:beg+prediction[i].shape[0]])
        #     beg = beg+prediction[i].shape[0]
        #     results.append(res)
        # result = torch.sum(torch.tensor(results))/len(results)
        # print(result)
        # if show:
        #     print('pred: ',prediction[0].view(-1))
        #     print('real: ',target[0:prediction[0].shape[0]].view(-1))
        valid_indices = (target != -1)
        target = target[valid_indices].view(-1)
        # Calculate the starting indices of each batch
        batch_sizes = [pred.shape[0] for pred in prediction]
        batch_start_indices = [0] + batch_sizes[:-1]
        batch_start_indices = torch.cumsum(torch.tensor(batch_start_indices), dim=0)
        # Get corresponding target slices
        target_slices = [target[start_idx:start_idx + size] for start_idx, size in zip(batch_start_indices, batch_sizes)]
        # Calculate the cross-entropy loss for each batch
        losses = torch.stack([F.cross_entropy(pred.view(pred.shape[0]), target_slice) for pred, target_slice in zip(prediction, target_slices)])
        result = losses.mean()
        return result
        
    def move_acc(self, prediction, target):
        beg,move_accs,valid_indices,best_acc = 0,[], (target != -1),[]
        target = target[valid_indices].view(-1)
        for i in range(len(prediction)):
            # prediction[i].view(-1).numpy().shape
            pred =torch.argmax(prediction[i].view(-1)).numpy()
            real = list(np.nonzero(target[beg:beg+prediction[i].shape[0]].view(-1).numpy())[0])
            reak_best = np.argmax(target[beg:beg+prediction[i].shape[0]].view(-1).numpy())
            best_acc.append(1 if pred == reak_best else 0)
            move_accs.append(1 if pred in real else 0)
            beg = beg+prediction[i].shape[0]
        result = np.sum(move_accs)/len(move_accs)
        best_acc_result = np.sum(best_acc)/len(best_acc)
        return result,best_acc_result

    def training_step(self, train_batch, batch_idx):
        if self.board_rep == 'graph' and self.policy_format == 'graph':
            if self.policy_format == 'graph':
                value, policy= self.forward(train_batch)
                t = train_batch[0].edge_attr[:,3:4]
                value_loss = self.mse_loss(value, train_batch[1].view(value.shape))
                policy_loss = self.cross_entropy(policy , t)
                # print('real value: ', train_batch[1].view(value.shape))
                # print('predicted value: ', value)
                # print('Real, Pred: ')
                # print([(float(train_batch[1][i]),float(value[i])) for i in range(value.shape[0])])
                # print('loss: ', float(loss))
                move_acc,best_acc = self.move_acc(policy, t)
                if np.random.uniform() < 0.5:
                    print('pred: ', policy[0].view(-1))
                    t = t[(t!=-1)].view(-1)
                    print('real: ', t[0:policy[0].shape[0]].view(-1))
        else:
            value, policy= self.forward(train_batch)
            target_value = train_batch[1].view(value.shape)
            target_policy = train_batch[2]
            target_policy = target_policy.view(target_policy.size(0), -1) 
            move_acc,best_acc = array_move_acc(policy, target_policy)
            value_loss,policy_loss = self.alpha_loss(target_value, value, target_policy, policy)
        print('move acc: ', move_acc)
        print('best acc: ', best_acc)
        print('Value Loss: ', float(value_loss))
        print('Policy Loss: ', float(policy_loss))
        loss = policy_loss + value_loss
        self.log('train_loss', loss)
        self.log('move_acc', move_acc)
        self.log('best_acc', best_acc)
        self.log('MSE (value)', value_loss)
        self.log('Cross Entropy (policy)', policy_loss)
        return loss

    def test_step(self, test_batch,batch_idx):
        if self.board_rep == 'graph' and self.policy_format == 'graph':
            value, policy= self.forward(test_batch)
            value_loss = self.mse_loss(value, test_batch[1].view(value.shape))
            t = test_batch[0].edge_attr[:,3:4]
            policy_loss = self.cross_entropy(policy , t)
            # test if pred move is in any of the non zero idxs of real policy
            acc,best_move_acc = self.move_acc(policy, test_batch[0].edge_attr[:,3:4])
        else:
            value, policy= self.forward(test_batch)
            target_value = test_batch[1].view(value.shape)
            target_policy = test_batch[2]
            target_policy = target_policy.view(target_policy.size(0), -1) 
            acc,best_move_acc = array_move_acc(policy, target_policy)
            value_loss,policy_loss = self.alpha_loss(target_value, value, target_policy, policy)
        self.log_dict({'cross_entropy':policy_loss,'mse':float(value_loss),'move_acc':acc,'best_move_acc':best_move_acc},batch_size=1)
        return value_loss,policy_loss,acc, best_move_acc


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,weight_decay=0.0001)
        return optimizer

if __name__ == '__main__':

    ##### Testing #####
    from torch_geometric.loader import DataLoader
    from datamodule import ChessDataset, ChessDataset_arrays
    import chess
    config = {'lr': 0.0001,'board_representation':'array'}
    # params = 'networks/graph_31k2100k_params'
    temp = ChessDataset_arrays(boards=[chess.Board() for _ in range(64)],values=torch.tensor([0 for _ in range(64)]).to(torch.float),policies=[torch.tensor(np.random.randint(low=0,high=1,size=(4672,1))).view(8,8,73) for _ in range(64)])
    temp_dl = DataLoader(temp, batch_size=32, shuffle=True, num_workers=0)
    model = Network(config)
    # model.load_state_dict(torch.load(params))
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=10000)
    from torchsummary import summary
    summary(model,(8,8,21))
    trainer.fit(model, temp_dl)