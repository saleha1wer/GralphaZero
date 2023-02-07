import pandas as pd
from datamodule import ChessDataset
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from network import GNN

class Buffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = pd.DataFrame(columns=['fen','value','policy'])

    def __len__(self):
        return len(self.buffer)

    def push(self, fens, values, policies):
        self.buffer = self.buffer.append(pd.DataFrame({'fen':fens,'value':values,'policy':policies}))
        if self.__len__() > self.max_size:
            diff = len(self.buffer.index) - self.max_size
            self.buffer.drop(index=self.buffer.index[:diff],inplace=True)

    def sample(self, batch_size):
        if batch_size > self.__len__():
            batch_size = self.__len__()
        rand_sample = self.buffer.sample(n=batch_size)
        fens = rand_sample['fen'].tolist()
        values = rand_sample['value'].tolist()
        policies = rand_sample['policy'].tolist()
        data = ChessDataset(fens=fens,values=torch.tensor(values).to(torch.float),policies=torch.tensor(policies).to(torch.float))
        return data


##### Testing #####
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# from network import GNN
# import pytorch_lightning as pl
# temp = Buffer(100)
# temp.push(['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1','rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],[0,0],[np.random.uniform(size=(8,8,73)),np.random.uniform(size=(8,8,73))])
# data = temp.sample(2)
# config = {'lr': 0.001, 'hidden': 4672, 'n_layers': 8, 'batch_size': 2}
# temp_dl = DataLoader(data, batch_size=config['batch_size'], shuffle=True, num_workers=0)
# model = GNN(config)
# trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=10)
# trainer.fit(model, temp_dl)