"""
Training script for human games.
"""
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from load_data import load_data
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import torch
from network import GNN
import pickle
from utils.action_encoding import decode_action
import time 

def train_human_games(network, data_path,bufferpath=None,network_path=None,batch_size=64, epochs=20,save_name='network_human_games',testing_range=(0,1000)):
    if bufferpath is None:
        print('Loading data from: ', data_path)
        print('Range: ', testing_range)
        buffer = load_data(data_path, testing=True, testing_range=testing_range)
    else:
        with open(bufferpath, "rb") as f:
            buffer = pickle.load(f)
        print('Loaded buffer from: ', bufferpath)
        print('Buffer length: ', buffer.__len__())
    def test():
        # import pandas as pd
        # import numpy as np
        # pd.set_option('display.max_rows', None)
        # print(buffer.buffer['value'])
        # count = 0
        # for index, row in buffer.buffer.iterrows():
        #     print(row['board'].fen())
        #     move_idx = np.unravel_index(row['policy'].argmax(), row['policy'].shape)
        #     print(move_idx)
        #     print(decode_action(row['board'], row['policy']))
        #     count += 1
        #     if count > 20:
                raise

    data = buffer.sample_all()
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(accelerator=device, devices=1, max_epochs=epochs)
    if network_path is not None:
        start = time.time()
        print('Loading Network from: ', network_path)
        network = torch.load(network_path)
        print('Loaded network from: ', network_path)
        print('Time taken: ', time.time() - start)
    network.train()
    trainer.fit(network, dataloader)
    torch.save(network.state_dict(), save_name)

if __name__ == '__main__':
    data_path = '/Users/mrsalwer/Desktop/graph_chess/train_human_games/human_games/games_chesstempo.csv'
    bufferpath = '0_5000_games_buffer'
    # bufferpath = None
    testing_range = (0,100)
    network_path = None
    gnn_type = 'GraphSAGE'
    # gnn_type = 'GAT'
    network = GNN({'lr': 0.0001 , 'hidden': 4672, 'n_layers': 1,'heads': 4,'gnn_type':gnn_type})
    save_name = 'network_human_games_params'
    train_human_games(network, data_path,bufferpath=bufferpath, network_path=network_path,batch_size=32, epochs=1,save_name=save_name,testing_range=testing_range)
    #from saved buffer train


