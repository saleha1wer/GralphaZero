# load human games 
from load_data import load_data
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from network import Network
from old_network import Network as oldNetwork
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as torchDataLoader
import pytorch_lightning as pl
import torch
import copy
import asyncio
from array_net import AlphaLoss
from array_net import move_acc as array_move_acc
import numpy as np
def getstrbet(s, start, end):
    return s[s.index(start) + len(start):s.rindex(end)]
def test_human(configs,params,num_test_pos,range):
    data_path = '/Users/mrsalwer/Desktop/graph_chess/train_human_games/human_games/games_chesstempo.csv'


    nets = []
    names = []
    for config,param in zip(configs,params):
        net = Network(config)
        net.load_state_dict(torch.load(param))
        nets.append(copy.deepcopy(net))
        if config['board_representation'] == 'array':
            name = 'Array ({})'.format(getstrbet(param,'array_','_params'))
        else:
            if config['policy_format'] == 'array':
                name = 'Graph (array policy) ({})'.format(getstrbet(param,'graph_','_params'))
            else:
                name = 'Graph ({})'.format(getstrbet(param,'graph_','_params')) 
        names.append(name)
    policy_format = 'both' if ([config['policy_format'] for config in configs].count('array') > 0 and [config['policy_format'] for config in configs].count('graph') > 0) else configs[0]['policy_format']
    buf = asyncio.run(load_data(path=data_path,
                    testing=True,
                    stockfish_value=True,
                    testing_range=range,
                    time_limit=0.01,
                    n_random_pos=num_test_pos,
                    policy_format=policy_format))

    results = []
    batch_size=min(64,int(num_test_pos/4))
    for idx,net in enumerate(nets):
        net.eval()
        trainer = pl.Trainer(accelerator='cpu', devices=1)
        print('Testing Network: ',names[idx])
        if net.policy_format == 'array':
            data = buf.sample_all(dataset_type=net.board_rep,policy_format=net.policy_format,return_boards=True)
            dataloader = torchDataLoader(data, batch_size=batch_size,drop_last=True) if net.board_rep == 'array' else DataLoader(data, batch_size=min(1024,int(num_test_pos/4)),drop_last=True)
            alpha_loss = AlphaLoss()
            res = trainer.predict(net,dataloader)
            total_mse = 0
            total_cse = 0
            for i, batch_res in enumerate(res):
                pred_value, pred_policy = batch_res
                start, stop =i * dataloader.batch_size,(i+1) * dataloader.batch_size
                true_value = data.values[start : stop]
                true_policy = np.stack(data.policies[start: stop])
                true_policy = torch.from_numpy(true_policy)
                true_value = torch.from_numpy(true_value)
                true_policy = true_policy.view(true_policy.size(0), -1) 
                pred_value = pred_value.view(pred_value.size(0), -1)
                pred_policy = pred_policy.view(pred_policy.size(0), -1)
                true_value = true_value.view(true_value.size(0), -1)
                if pred_value.shape != true_value.shape:
                    print(pred_value)
                    print('  ')
                    print('  ')
                    print(true_value)
                    print(pred_value.shape,true_value.shape)
                assert (pred_value.shape == true_value.shape) and (pred_policy.shape == true_policy.shape)
                mse,cse = alpha_loss(true_value, pred_value, true_policy, pred_policy)
                move_acc, best_move_acc = array_move_acc(pred_policy, true_policy, mask_illegal=True, boards=data.boards[start:stop])
                total_mse += mse
                total_cse += cse
            total_mse /= len(res)
            total_cse /= len(res)
            results.append({'mse':total_mse,'cse':total_cse,'move_acc':move_acc,'best_move_acc':best_move_acc})
        else:
            data = buf.sample_all(dataset_type=net.board_rep,policy_format=net.policy_format)
            dataloader = DataLoader(data, batch_size=batch_size,drop_last=True) 
            res = trainer.test(net,dataloader)
            results.append(res)

    for idx,res in enumerate(results):
        print('Network: ',names[idx],'\n')
        print('Score:', res,'\n \n')

if __name__ == '__main__':
    params_graph ='networks/graph_graph_35k2800k_params'
    config_graph = {'board_representation':'graph','policy_format':'graph','lr': 0.00003,'GAheads': 16, 'att_emb_size': 1024, 'heads_GAT_edge': 16, 'heads_GAT_graph': 256, 'hidden_edge': 2048, 'hidden_graph': 512, 'n_layers': 5, 'pol_nlayers': 5, 'value_nlayers': 5,'finetune':False}

    params_array = 'networks/array_array_35k2800k_params'
    config_array = {'lr': 0.00001,'board_representation':'array','policy_format':'array','finetune':False}

    params_graph_array = 'networks/graph_array_35k2800k_params'
    config_graph_array = {'board_representation':'graph','policy_format':'array','finetune':False,'lr': 0.00001, 'heads_GAT_graph': 64, 'hidden_graph': 2048, 'n_layers': 5, 'pol_nlayers': 5, 'value_nlayers': 5}

    params = [params_array,params_graph,params_graph_array]
    configs = [config_array,config_graph,config_graph_array]
    
    # params = [params_graph_array,params_array]
    # configs = [config_graph_array,config_array]
    
    
    test_human(configs,params,4000,(0,50))
    