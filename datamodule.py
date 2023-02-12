import pandas as pd
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Dataset
import chess

from utils.board2graph import board2graph

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class ChessDataset(Dataset):
    def __init__(self,boards,values,policies):
        """
        boards = List of chess boards
        values = List of values
        policies = List of policies
        """
        self.boards = boards
        self.values = np.array(values)
        self.policies = np.array(policies)
        self.graphs = [board2graph(i) for i in boards]
        self.length = len(self.boards)
        super().__init__('data', None, None, None)

    @property
    def raw_file_names(self):
        return ['temp']

    @property
    def processed_file_names(self):
        return ['temp.pt']

    def process(self):
        pass

    def len(self):
        return self.length

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        # temp = self.data.iloc[idx]
        # return torch.tensor(temp['graph']), torch.tensor(temp['value']), torch.tensor(temp['policy'])
        return self.graphs[idx], torch.tensor(self.values[idx]), torch.tensor(self.policies[idx])



##### Testing #####
# temp = ChessDataset(fens=['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1','rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],values=[0,0],policies=[[0,0,0],[0,0,0]])
# temp.process()
# datamodule_config = {
#     'batch_size': 2,
#     'num_workers': 0
# }
# temp_dl = DataLoader(temp, batch_size=datamodule_config['batch_size'], shuffle=True, num_workers=datamodule_config['num_workers'])
# print(next(iter(temp_dl)))
