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

pd.set_option('display.width', 100000)

def train_human_games(network,board_rep, data_path,trainer,policy_format=None,bufferpath=None,buffer=None,network_path=None,batch_size=64, epochs=20,save_name='network_human_games',testing_range=(0,1000)):
    if buffer is not None:
        pass
    elif bufferpath is None:
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
        #     print(decode_action(row['board'], row['policy'],exploration=False))
        #     count += 1
        #     if count > 20:
                raise

    data = buffer.sample_all(type=board_rep)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True,drop_last=True) if board_rep == 'graph' else torchDataLoader(data, batch_size=batch_size, shuffle=True,drop_last=True)
    policy_format = policy_format if policy_format is not None else board_rep
    # Get current date and time
    if network_path is not None:
        start = time.time()
        print('Loading Network from: ', network_path)
        network.load_state_dict(torch.load(network_path))
        print('Loaded network from: ', network_path)
        print('Time taken: ', time.time() - start)
    network.train()
    trainer.fit(network, dataloader)
    trainer.save_checkpoint('checkpoints/checkpoint.ckpt')
    torch.save(network.state_dict(), save_name)


def train_human(configs, n_human_games, n_random_pos,data_path,device,interval_len=1500,batch_size=256):
    ranges = [(i, i+interval_len) for i in range(0, n_human_games, interval_len)]
    networks = [Network(config) for config in configs]
    loggers = [TensorBoardLogger("tb_logs", name=board_rep+'_'+policy_format) for board_rep,policy_format in zip([config['board_representation'] for config in configs],[config['policy_format'] for config in configs])]
    log_steps = 1
    for idx,testing_range in enumerate(ranges):
        print('Range:', testing_range)
        num_h_games = str(testing_range[1])[:1] if testing_range[1] < 10000 else str(testing_range[1])[:2]
        save_names = ['networks/{}_{}k{}k_params'.format(board_rep+'_'+policy_format,num_h_games,(n_random_pos//1000)*(idx+1)) for board_rep,policy_format in zip([config['board_representation'] for config in configs],[config['policy_format'] for config in configs])]
        print('Will save to \n', save_names)
        # save_names = ['TEMP' for _ in range(len(networks))]
        policy_format = 'both' if ([config['policy_format'] for config in configs].count('array') > 0 and [config['policy_format'] for config in configs].count('graph') > 0) else configs[0]['policy_format']
        buffer = load_data(path=data_path,
                policy_format=policy_format,
                testing=True,
                stockfish_value=True,
                testing_range=testing_range,
                save_path=None,
                time_limit=0.01,
                n_random_pos=n_random_pos)
        for config,network,save_name,logger in zip(configs,networks,save_names,loggers):
            trainer = pl.Trainer(accelerator=device, devices=1, max_epochs=1+idx,logger=logger,log_every_n_steps=log_steps)
            data = buffer.sample_all(dataset_type=config['board_representation'],policy_format=config['policy_format'])
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=True,drop_last=True) if config['board_representation'] == 'graph' else torchDataLoader(data, batch_size=batch_size, shuffle=True,drop_last=True)
            network.train()
            if idx > 0:
                trainer.fit(network, dataloader,ckpt_path='checkpoints/checkpoint_{}_{}.ckpt'.format(config['board_representation'],config['policy_format']))
            else:
                trainer.fit(network, dataloader)
            trainer.save_checkpoint('checkpoints/checkpoint_{}_{}.ckpt'.format(config['board_representation'],config['policy_format']))
            torch.save(network.state_dict(), save_name)
        del buffer
    pass

if __name__ == '__main__':
    data_path = '/Users/mrsalwer/Desktop/graph_chess/train_human_games/human_games/games_chesstempo.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_graph_graph = {'board_representation':'graph','policy_format':'graph','value_nlayers':10,'value_hidden':256,'pol_nlayers':10,'pol_hidden':256,'dropout':0,'hidden_graph':256,'hidden_edge':512,'residual':True,'att_emb_size':256,'GAheads':32,'loss_func':None,'lr': 0.0001, 'hidden': 2048, 'n_layers': 3,'heads': 32}
    config_graph_array = {'board_representation':'graph','policy_format':'array','value_nlayers':10,'value_hidden':256,'pol_nlayers':10,'pol_hidden':256,'dropout':0,'hidden_graph':256,'hidden_edge':512,'residual':True,'att_emb_size':256,'GAheads':32,'loss_func':None,'lr': 0.0001, 'hidden': 2048, 'n_layers': 3,'heads': 32}
    config_array_array = {'lr': 0.0001,'board_representation':'array','policy_format':'array'}
    configs = [config_graph_array,config_graph_graph,config_array_array]
    total_human_games,n_random_pos,interval_len,batch_size = 50000,10000,150,256
    # total_human_games,n_random_pos,interval_len,batch_size = 50000,10000,150,256
    # total_human_games,n_random_pos,interval_len,batch_size = 50000,100000,1500,256
    train_human(configs,n_human_games=total_human_games,n_random_pos=n_random_pos,data_path=data_path,device=device,interval_len=interval_len,batch_size=batch_size)

if __name__ == '__main__' and False:
    data_path = '/Users/mrsalwer/Desktop/graph_chess/train_human_games/human_games/games_chesstempo.csv'
    ranges = [(i, i+1500) for i in range(12000, 30000, 1500)]
    # board_rep,gnn_type,useresnet = 'array',None, False
    # config = {'value_nlayers':5,'value_hidden':256,'pol_nlayers':5,'pol_hidden':256,'dropout':0,'hidden_graph':256,'hidden_edge':512,'residual':True,'att_emb_size':256,'GAheads':32,'loss_func':None,'lr': 0.0001, 'hidden': 2048, 'n_layers': 3,'heads': 32,'board_representation':'graph','useresnet':False}
    config = {'lr': 0.0001,'board_representation':'array','policy_format':'array'}
    network = Network(config)
    # network.load_state_dict(torch.load('new_networks/graph_12k800k_params'))
    n_thousand_rand_pos = 100
    board_rep = config['board_representation']
    policy_format = board_rep
    logger = TensorBoardLogger("tb_logs", name=board_rep+'_'+policy_format)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for idx,testing_range in enumerate(ranges):
        testing_range = (0,5)
        print('Range:', testing_range)
        num_h_games = str(testing_range[1])[:1] if testing_range[1] < 10000 else str(testing_range[1])[:2]
        # save_name = 'new_networks/{}_{}k{}k_params'.format(board_rep,num_h_games,n_thousand_rand_pos*(idx+1)+800)
        save_name = 'TEMP'
        print('Will save to ', save_name)
        buf = load_data(path=data_path,
                policy_format=config['board_representation'],
                testing=True,
                stockfish_value=True,
                testing_range=testing_range,
                save_path=None,
                time_limit=0.01,
                n_random_pos=n_thousand_rand_pos*1)
        batch_size = 2
        log_steps = 10
        if idx == 0:
            trainer = pl.Trainer(accelerator=device, devices=1, max_epochs=1,logger=logger,log_every_n_steps=log_steps)
        else:
            trainer = pl.Trainer(accelerator=device, devices=1, max_epochs=1,logger=logger,log_every_n_steps=1,resume_from_checkpoint='checkpoints/checkpoint.ckpt')
        train_human_games(network,config['board_representation'], data_path,trainer,bufferpath=None,buffer=buf, network_path=None,batch_size=batch_size, epochs=1,save_name=save_name,testing_range=testing_range)
        def masked_acc():
            from utils.board2array import board2array
            from utils.action_encoding import old_encode_action
            from load_data import get_stockfish_value_moves
            network.eval()
            fens = ['5K1B/1pP5/bp1pp2p/Rprr1P2/p5Pq/P2P4/p4R1P/1NNnk1b1 w - - 0 1','b7/1pP1k1pn/pP2n1rR/1P2p3/1p5P/PK1p4/Pprb3Q/1BBN3R w - - 0 1','2B5/PP2pp1n/1b3N2/1Pk1Brp1/2p1rp1Q/q2b1PpK/P1P1np2/4N3 w - - 0 1','Q1b5/2B2P2/4p2k/2K2p2/3p3P/1Ppp1p2/5N2/4n3 w - - 0 1','4r3/3p2pp/1P6/3P1K2/5P2/Bb2p1k1/1RP1B1p1/8 w - - 0 1','2B5/2p5/P1P3p1/2K2p2/8/p4PPN/2pk1B2/r2q4 w - - 0 1','8/p2PP1p1/k1P3B1/3p4/3rp1nP/1K1N4/1p4p1/8 w - - 0 1','3K4/6n1/1p6/Pb2B3/6Q1/p3P1P1/Rp1p1P1n/1k6 w - - 0 1','3N4/p1PPp3/7p/5KbP/3p2pN/k5P1/4BP2/8 w - - 0 1','K4Q2/3p2P1/1B6/P2pq3/nb6/1p2pR2/8/2k2Bn1 w - - 0 1','3k2Kb/QP6/1P1P1p2/1pp2N2/p5p1/6n1/4N1P1/8 w - - 0 1']
            boards = [chess.Board(fen) for fen in fens]
            boards = torch.stack([torch.tensor(board2array(board)) for board in boards])
            preds = network([boards])
            acc = []
            for fen,pred in zip(fens,preds[1]):
                board = chess.Board(fen)
                pred = pred.detach().numpy().reshape(8,8,73)
                legal_moves_idx = np.array([np.array(old_encode_action(board,i)) for i in board.legal_moves])
                new_pred = np.zeros((8,8,73))
                new_pred.fill(-np.inf)
                new_pred[legal_moves_idx[:,0],legal_moves_idx[:,1],legal_moves_idx[:,2]] = pred[legal_moves_idx[:,0],legal_moves_idx[:,1],legal_moves_idx[:,2]]
                _,best_moves = get_stockfish_value_moves([board],'array',move_lists=True)
                best_moves = [old_encode_action(board,m) for m in best_moves[0]]
                pred = list(np.unravel_index(np.argmax(new_pred),(8,8,73)))
                v = 1 if pred in best_moves else 0
                acc.append(v)
            print('Accuracy: ', np.mean(acc))
        del buf
