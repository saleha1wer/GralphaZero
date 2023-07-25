import pandas as pd
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Dataset
from torch.utils.data import Dataset as torchDataset
import chess
from utils.board2array import board2array
from utils.board2graph import board2graph

# print(f"Torch version: {torch.__version__}")
# print(f"Cuda available: {torch.cuda.is_available()}")
# print(f"Torch geometric version: {torch_geometric.__version__}")



class ChessDataset(Dataset):
    def __init__(self,boards,values,policies,log=True):
        """
        boards = List of chess boards
        values = List of values
        policies = List of policies
        """
        self.boards = boards
        self.values = np.array(values)
        self.policies = policies # list of tensors (that contain policy values for each graph)
        if len(policies[0].shape) < 2:
            self.graphs = [board2graph(b,policies[idx]) for idx,b in enumerate(boards)]
            self.policy_format = 'graph'
        else:
            self.graphs = [board2graph(b) for b in boards]
            self.policy_format = 'array'
        self.length = len(self.boards)
        super().__init__('data', None, None, None,log=log)

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
        if self.policy_format == 'graph':
            return self.graphs[idx], torch.tensor(self.values[idx]) # policy is encoded in the graph
        elif self.policy_format == 'array':
            return self.graphs[idx], torch.tensor(self.values[idx]), self.policies[idx]


class oldChessDataset(Dataset):
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

class ChessDataset_arrays(torchDataset):
    def __init__(self,boards,values,policies,transform=None, target_transform=None,return_boards=False):
        self.boards = boards
        self.values = np.array(values)
        self.policies = torch.stack(policies)
        self.arrays = [board2array(i) for i in boards]
        self.length = len(self.boards)
        self.transform = transform
        self.target_transform = target_transform
        self.return_boards = return_boards
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.return_boards:
            return  torch.tensor(self.arrays[idx]), torch.tensor(self.values[idx]), self.policies[idx],idx
        else:
            return  torch.tensor(self.arrays[idx]), torch.tensor(self.values[idx]), self.policies[idx]

##### Testing #####
# temp = ChessDataset(fens=['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1','rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],values=[0,0],policies=[[0,0,0],[0,0,0]])
# temp.process()
# datamodule_config = {
#     'batch_size': 2,
#     'num_workers': 0
# }
# temp_dl = DataLoader(temp, batch_size=datamodule_config['batch_size'], shuffle=True, num_workers=datamodule_config['num_workers'])
# print(next(iter(temp_dl)))
