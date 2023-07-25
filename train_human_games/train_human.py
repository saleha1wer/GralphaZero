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
from torch.utils.data import DataLoader as torchDataLoader
import torch
from network import Network
import pickle
from utils.action_encoding import decode_action
import time 
from buffer import Buffer
import chess
import numpy as np
import pandas as pd
from torchsummary import summary
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import asyncio

pd.set_option('display.width', 100000)

def train_human(configs, n_human_games, n_random_pos,data_path,device,params_path=None,interval_len=1500,batch_size=256):
    ranges = [(i, i+interval_len) for i in range(30000, n_human_games, interval_len)]
    networks = [Network(config) for config in configs]
    if params_path is not None:
        for i,path in enumerate(params_path):
            networks[i].load_state_dict(torch.load(path))
    loggers = [TensorBoardLogger("tb_logs", name=board_rep+'_'+policy_format,version=0) for board_rep,policy_format in zip([config['board_representation'] for config in configs],[config['policy_format'] for config in configs])]
    log_steps = 25
    for idx,testing_range in enumerate(ranges):
        print('Range:', testing_range)
        if testing_range[1] < 10000:
            num_h_games = str(testing_range[1])[:1]
        else:
            num_h_games = str(testing_range[1])[:2] if testing_range[1] < 100000 else str(testing_range[1])[:3]
        save_names = ['networks/{}_{}k{}k_params'.format(board_rep+'_'+policy_format,num_h_games,(n_random_pos//1000)*(idx+1)+2400) for board_rep,policy_format in zip([config['board_representation'] for config in configs],[config['policy_format'] for config in configs])]
        print('Will save to \n', save_names)
        # save_names = ['TEMP' for _ in range(len(networks))]
        policy_format = 'both' if ([config['policy_format'] for config in configs].count('array') > 0 and [config['policy_format'] for config in configs].count('graph') > 0) else configs[0]['policy_format']
        buffer =asyncio.run(load_data(path=data_path,
                policy_format=policy_format,
                testing=True,
                stockfish_value=True,
                testing_range=testing_range,
                save_path=None,
                time_limit=0.01,
                n_random_pos=n_random_pos))
        for config,network,save_name,logger in zip(configs,networks,save_names,loggers):
            trainer = pl.Trainer(accelerator=device, devices=1, max_epochs=1+idx+6,logger=logger,log_every_n_steps=log_steps)
            data = buffer.sample_all(dataset_type=config['board_representation'],policy_format=config['policy_format'])
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=True,drop_last=True) if config['board_representation'] == 'graph' else torchDataLoader(data, batch_size=batch_size, shuffle=True,drop_last=True)
            network.train()
            if idx > -1:
                trainer.fit(network, dataloader,ckpt_path='checkpoints/checkpoint_{}_{}.ckpt'.format(config['board_representation'],config['policy_format']))
            else:
                trainer.fit(network, dataloader)
            trainer.save_checkpoint('checkpoints/checkpoint_{}_{}.ckpt'.format(config['board_representation'],config['policy_format']))
            torch.save(network.state_dict(), save_name)
            del data
            del dataloader
        del buffer

if __name__ == '__main__':
    data_path = '/Users/mrsalwer/Desktop/graph_chess/train_human_games/human_games/games_chesstempo.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_graph_graph = {'board_representation':'graph','policy_format':'graph','finetune':False,'lr': 0.00001,'GAheads': 16, 'att_emb_size': 1024, 'heads_GAT_edge': 16, 'heads_GAT_graph': 256, 'hidden_edge': 2048, 'hidden_graph': 512, 'n_layers': 5, 'pol_nlayers': 5, 'value_nlayers': 5}
    # config_graph_array = {'board_representation':'graph','policy_format':'array','finetune':False,'value_nlayers':10,'pol_nlayers':10,'hidden_graph':256,'lr': 0.0001, 'hidden': 2048, 'n_layers': 3,'heads': 32}
    # config_array_array = {'lr': 0.0001,'board_representation':'array','finetune':False,'policy_format':'array'}
    # configs = [config_graph_array,config_graph_graph,config_array_array]
    configs = [config_graph_graph]
    params_paths = ['networks/graph_graph_15k1200k_params']
    total_human_games,n_random_pos,interval_len,batch_size = 135000,100,5,64
    # total_human_games,n_random_pos,interval_len,batch_size = 135000,400000,5000,64
    train_human(configs,n_human_games=total_human_games,n_random_pos=n_random_pos,data_path=data_path,device=device,interval_len=interval_len,params_path=params_paths,batch_size=batch_size)
