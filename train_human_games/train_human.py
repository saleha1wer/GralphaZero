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

def train_human_games(network, data_path,batch_size=64, epochs=20):
    data_path = '/Users/mrsalwer/Desktop/graph_chess/train_human_games/human_games/games_chesstempo.csv'
    buffer = load_data(data_path, testing=False)
    data = buffer.sample_all()
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(accelerator=device, devices=1, max_epochs=epochs)
    network.train()
    trainer.fit(network, dataloader)
    torch.save(network, 'network_human_games')

if __name__ == '__main__':
    data_path = '/Users/mrsalwer/Desktop/graph_chess/train_human_games/human_games/games_chesstempo.csv'
    network = GNN({'lr': 0.0005, 'hidden': 4672, 'n_layers': 4})
    train_human_games(network, data_path, batch_size=64, epochs=20)


