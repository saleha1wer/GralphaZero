import pandas as pd
from datamodule import ChessDataset
from torch_geometric.loader import DataLoader
import torch
import numpy as np
from network import GNN

class Buffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = pd.DataFrame(columns=['board','value','policy'])

    def __len__(self):
        return len(self.buffer.index)

    def push(self, boards, values, policies):
        self.buffer = pd.concat([self.buffer,pd.DataFrame({'board':boards,'value':values,'policy':policies})],ignore_index=True)
        if self.__len__() > self.max_size:
            diff = self.__len__() - self.max_size
            self.buffer = self.buffer.iloc[diff:]

    def push_df(self, df):
        self.buffer = pd.concat([self.buffer,df],ignore_index=True)
        if self.__len__() > self.max_size:
            diff = self.__len__() - self.max_size
            self.buffer = self.buffer.iloc[diff:]

    def sample(self, batch_size):
        if batch_size > self.__len__():
            batch_size = self.__len__()
        rand_sample = self.buffer.sample(n=batch_size)
        boards = rand_sample['board'].tolist()
        values = rand_sample['value'].tolist()
        policies = rand_sample['policy'].tolist()
        data = ChessDataset(boards=boards,values=torch.tensor(values).to(torch.float),policies=torch.tensor(policies).to(torch.float))
        return data

def join_buffers(buffers):
    buffer_df = pd.DataFrame(columns=['board','value','policy'])
    for i in buffers:
        buffer_df = pd.concat([buffer_df,i.buffer],ignore_index=True)
    return buffer_df

##### Testing #####

# from network import GNN
# import pytorch_lightning as pl
# import chess
# temp = Buffer(100)
# temp_board = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
# temp_board.push_uci('e2e4')
# temp_board.push_uci('e7e5')
# temp.push([temp_board,temp_board],[0,0],[np.random.uniform(size=(8,8,73)),np.random.uniform(size=(8,8,73))])
# config = {'lr': 0.001, 'hidden': 4672, 'n_layers': 8, 'batch_size': 2}

# temp2 = Buffer(100)
# temp2.push([temp_board,temp_board],[0,0],[np.random.uniform(size=(8,8,73)),np.random.uniform(size=(8,8,73))])
# temp3 = Buffer(100)
# temp3.push([temp_board,temp_board],[0,0],[np.random.uniform(size=(8,8,73)),np.random.uniform(size=(8,8,73))])

# new_buffer_df = join_buffers([temp,temp2,temp3])
# new_buffer = Buffer(100)
# new_buffer.push_df(new_buffer_df)
# data = new_buffer.sample(6)
# temp_dl = DataLoader(data, batch_size=config['batch_size'], shuffle=True, num_workers=0)
# model = GNN(config)
# trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=10)
# trainer.fit(model, temp_dl)