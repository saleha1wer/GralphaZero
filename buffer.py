import pandas as pd
from datamodule import ChessDataset, ChessDataset_arrays
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
import torch
import numpy as np
from old_network import Network
import pickle
import os
import chess

class Buffer:
    def __init__(self, max_size, policy_format='graph'):
        self.max_size = max_size
        self.policy_format = policy_format
        if policy_format == 'both':
            self.buffer = pd.DataFrame(columns=['board','value','graph_policy','array_policy'])
        else:
            self.buffer = pd.DataFrame(columns=['board','value','policy'])

    def __len__(self):
        return len(self.buffer.index)

    def push(self, boards, values, policies, graph_policies=None, array_policies=None):
        assert (graph_policies is None) == (array_policies is None) == (self.policy_format != 'both')
        if self.policy_format == 'both':
            graph_policies = [torch.from_numpy(pol).float() for pol in graph_policies]
            array_policies = [torch.from_numpy(pol).float() for pol in array_policies]
            self.buffer = pd.concat([self.buffer, pd.DataFrame({
                'board': boards,
                'value': values,
                'graph_policy': graph_policies,
                'array_policy': array_policies})], ignore_index=True)
        else:
            policies = [torch.from_numpy(pol).float() for pol in policies]
            self.buffer = pd.concat([self.buffer, pd.DataFrame({
                'board': boards,
                'value': values,
                'policy': policies})], ignore_index=True)
        if self.__len__() > self.max_size:
            diff = self.__len__() - self.max_size
            self.buffer = self.buffer.iloc[diff:]

    def push_df(self, df):
        self.buffer = pd.concat([self.buffer,df],ignore_index=True)
        if self.__len__() > self.max_size:
            diff = self.__len__() - self.max_size
            self.buffer = self.buffer.iloc[diff:]

    def sample(self, batch_size, dataset_type='graph', policy_format='graph'):
        if batch_size > self.__len__():
            batch_size = self.__len__()
        rand_sample = self.buffer.sample(n=batch_size)
        boards = rand_sample['board'].tolist()
        values = rand_sample['value'].tolist()
        boards, values = np.array(boards), np.array(values)

        if self.policy_format == 'both':
            graph_policies = rand_sample['graph_policy'].tolist()
            array_policies = rand_sample['array_policy'].tolist()
            if dataset_type == 'graph':
                data = ChessDataset(boards=boards, values=torch.tensor(values).to(torch.float), policies=graph_policies if policy_format == 'graph' else array_policies)
            else:
                data = ChessDataset_arrays(boards=boards, values=torch.tensor(values).to(torch.float), policies=graph_policies if policy_format == 'graph' else array_policies)
        else:
            policies = rand_sample['policy'].tolist()
            if dataset_type == 'graph':
                data = ChessDataset(boards=boards,values=torch.tensor(values).to(torch.float),policies=policies)
            else:
                data = ChessDataset_arrays(boards=boards,values=torch.tensor(values).to(torch.float),policies=policies)
        return data

    def sample_all(self, dataset_type='graph', policy_format='graph'):
        boards = self.buffer['board'].tolist()
        values = self.buffer['value'].tolist()
        boards, values = np.array(boards), np.array(values)

        if self.policy_format == 'both':
            graph_policies = self.buffer['graph_policy'].tolist()
            array_policies = self.buffer['array_policy'].tolist()
            if dataset_type == 'graph':
                data = ChessDataset(boards=boards, values=torch.tensor(values).to(torch.float), policies=graph_policies if policy_format == 'graph' else array_policies)
            else:
                data = ChessDataset_arrays(boards=boards, values=torch.tensor(values).to(torch.float), policies=graph_policies if policy_format == 'graph' else array_policies)
        else:
            policies = self.buffer['policy'].tolist()
            if dataset_type == 'graph':
                data = ChessDataset(boards=boards,values=torch.tensor(values).to(torch.float),policies=policies)
            else:
                data = ChessDataset_arrays(boards=boards,values=torch.tensor(values).to(torch.float),policies=policies)
        return data

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

def join_buffers(buffers):
    buffer_df = pd.DataFrame(columns=buffers[0].buffer.columns)
    for i in buffers:
        buffer_df = pd.concat([buffer_df,i.buffer],ignore_index=True)
    return buffer_df

class OldBuffer:
    def __init__(self, max_size,inc_preds=False,policy_format='graph'):
        self.inc_preds = inc_preds
        self.max_size = max_size
        if inc_preds:
            self.buffer = pd.DataFrame(columns=['board','value','policy','pred_value','pred_policy'])
        else:
            self.buffer = pd.DataFrame(columns=['board','value','policy'])

    def __len__(self):
        return len(self.buffer.index)

    def push(self, boards, values, policies,pred_values=None,pred_policies=None):
        assert (pred_values is None) == (pred_policies is None) == (not self.inc_preds)
        policies = [torch.from_numpy(pol).float() for pol in policies]
        if pred_policies is not None:
            pred_policies = [torch.from_numpy(pol).float() for pol in pred_policies]
        if self.inc_preds:
            self.buffer = pd.concat([self.buffer,pd.DataFrame({'board':boards,'value':values,'policy':policies,'pred_value':pred_values,'pred_policy':pred_policies})],ignore_index=True)
        else:
            self.buffer = pd.concat([self.buffer,pd.DataFrame({'board':boards,'value':values,'policy':policies})],ignore_index=True)
        if self.__len__() > self.max_size:
            diff = self.__len__() - self.max_size
            self.buffer = self.buffer.iloc[diff:]

    def push_df(self, df):
        self.buffer = pd.concat([self.buffer,df],ignore_index=True)
        if self.__len__() > self.max_size:
            diff = self.__len__() - self.max_size
            self.buffer = self.buffer.iloc[diff:]

    def sample(self, batch_size,type='graph'):
        if batch_size > self.__len__():
            batch_size = self.__len__()
        rand_sample = self.buffer.sample(n=batch_size)
        boards = rand_sample['board'].tolist()
        values = rand_sample['value'].tolist()
        policies = rand_sample['policy'].tolist()
        boards, values = np.array(boards), np.array(values)
        dataset = ChessDataset if type == 'graph' else ChessDataset_arrays

        if self.inc_preds:
            pred_values = rand_sample['pred_value'].tolist()
            pred_policies = rand_sample['pred_policy'].tolist()
            data = dataset(boards=boards,values=torch.tensor(values).to(torch.float),policies=policies,pred_values=torch.tensor(pred_values).to(torch.float),pred_policies=pred_policies)
        else:
            data = dataset(boards=boards,values=torch.tensor(values).to(torch.float),policies=policies)

        return data
    def sample_all(self,type='graph'):
        boards = self.buffer['board'].tolist()
        values = self.buffer['value'].tolist()
        policies = self.buffer['policy'].tolist()
        boards, values = np.array(boards), np.array(values)
        dataset = ChessDataset if type == 'graph' else ChessDataset_arrays

        if self.inc_preds:
            pred_values = self.buffer['pred_value'].tolist()
            pred_policies = self.buffer['pred_policy'].tolist()
            data = dataset(boards=boards,values=torch.tensor(values).to(torch.float),policies=policies,pred_values=torch.tensor(pred_values).to(torch.float),pred_policies=pred_policies)
        else:
            data = dataset(boards=boards,values=torch.tensor(values).to(torch.float),policies=policies)
        return data

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    # def load(self, path):
    #     for file in ['board','value','policy']:
    #         with open(os.path.join(path,file), "rb") as f:
    #             self.buffer[file] = pickle.load(f)
        

##### Testing #####

if __name__ == '__main__':
    # from network import Network
    import pytorch_lightning as pl
    # import chess
    # with open('0_5000_games_buffer', "rb") as f:
    #     temp = pickle.load(f)
    # temp.save('0_5000_buffer')
    config = {'lr': 0.0003 , 'hidden': 4672, 'n_layers': 1,'heads': 64,'gnn_type':None,'board_representation':'array','useresnet':False}
    temp2 = Buffer(100)
    temp_board = chess.Board()
    temp2.push([temp_board,temp_board],[0,0],[np.random.uniform(size=(8,8,73)),np.random.uniform(size=(8,8,73))],[np.random.uniform(size=(12)),np.random.uniform(size=(12))],[np.random.uniform(size=(8,8)),np.random.uniform(size=(8,8))])
    # temp3 = Buffer(100)
    # temp3.push([temp_board,temp_board],[0,0],[np.random.uniform(size=(8,8,73)),np.random.uniform(size=(8,8,73))])

    # new_buffer_df = join_buffers([temp,temp2,temp3])
    # new_buffer = Buffer(100)
    # new_buffer.push_df(new_buffer_df)
    data = temp2.sample(6,type='array')
    temp_dl = DataLoader(data, batch_size=2, shuffle=True, num_workers=0)
    model = Network(config)
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=10)
    trainer.fit(model, temp_dl)
    