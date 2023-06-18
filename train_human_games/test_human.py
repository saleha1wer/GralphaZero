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
def getstrbet(s, start, end):
    return s[s.index(start) + len(start):s.rindex(end)]
def test_human(configs,params,num_test_pos):
    data_path = '/Users/mrsalwer/Desktop/graph_chess/train_human_games/human_games/games_chesstempo.csv'


    nets = []
    names = []
    for config,param in zip(configs,params):
        net = Network(config)
        net.load_state_dict(torch.load(param))
        nets.append(copy.deepcopy(net))
        name = 'Graph ({})'.format(getstrbet(param,'graph_','_params')) 
        name = 'old ' + name if 'new_g' in param else name
        names.append(name)

    buf = load_data(path=data_path,
                    testing=True,
                    stockfish_value=True,
                    testing_range=(0,0),
                    time_limit=0.01,
                    n_random_pos=num_test_pos,
                    enc_all_legal = False)
    data = buf.sample_all(type='graph')
    dataloader = DataLoader(data, batch_size=min(1024,num_test_pos),drop_last=True) 
    # torchdata = buf.sample_all(type='array')
    # torchdataloader = torchDataLoader(torchdata, batch_size=32,drop_last=True)
    results = []
    for idx,net in enumerate(nets):
        net.eval()
        trainer = pl.Trainer(accelerator='cpu', devices=1)
        print('Testing Network: ',names[idx])
        res = trainer.test(net,dataloader)
        results.append(res)

    for idx,res in enumerate(results):
        print('Network: ',names[idx],'\n')
        print('Acc:', res,'\n \n')
if __name__ == '__main__':
    params1 = 'new_networks/new_graph_30k2000k_params'
    params2 = 'new_networks/old_graph_33k2200k_params'
    # params3 = 'new_networks/new_graph_6k400k_params'
    # params4 = 'new_networks/new_graph_7k500k_params'
    # config2 = {'lr': 0.0001 , 'hidden': 4672, 'n_layers': 2,'heads': 16,'gnn_type':'GAT','board_representation':'graph','useresnet':False}
    # params5 = 'new_networks/new_graph_9k600k_params'
    # config3 = {'lr': 0.0001 , 'hidden': 4672, 'n_layers': 2,'heads': 16,'gnn_type':'GAT','board_representation':'array','useresnet':False}
    # params6 = 'new_networks/new_graph_10k700k_params'
    # params7 = 'new_networks/new_graph_12k800k_params'
    # params8 = 'new_networks/new_graph_13k900k_params'
    # params9 = 'new_networks/new_graph_15k1000k_params'
    # params10 = 'new_networks/new_graph_16k1100k_params'
    # params11 = 'new_networks/new_graph_18k1200k_params'
    # params12 = 'new_networks/new_graph_19k1300k_params'
    # params13 = 'new_networks/new_graph_21k1400k_params'
    # params14 = 'new_networks/new_graph_22k1500k_params'
    # params15 = 'new_networks/new_graph_27k1800k_params'
    # params3 = 'new_networks/new_graph_36k2400k_params'
    # params4 = 'new_networks/new_graph_40k2700k_params'
    # params5 = 'new_networks/new_graph_45k3000k_params'
    # params6 = 'new_networks/new_graph_48k3200k_params'
    # params7 = 'new_networks/new_graph_51k3400k_params'
    # params8 = 'new_networks/new_graph_52k3500k_params'
    # params9 = 'new_networks/new_graph_55k3700k_params'
    # params10 = 'new_networks/new_graph_57k3800k_params'
    # params11 = 'new_networks/new_graph_60k4000k_params'
    # params12 = 'new_networks/new_graph_60k4000k_params_vs_stockloop0'
    # params13 = 'new_networks/new_graph_60k4000k_params_vs_stockloop4'
    # params14 = 'new_networks/new_graph_60k4000k_params_vs_stockloop8'
    # params14 = 'new_networks/new_graph_60k4000k_params_vs_stockloop10'
    # params15 = 'new_networks/new_graph_60k4000k_params_vs_stockloop12'
    # params16 = 'new_networks/new_graph_60k4000k_params_vs_stockloop14'
    # params17 = 'new_networks/new_graph_60k4000k_params_vs_stockloop16'
    # params18 = 'new_networks/new_graph_60k4000k_params_vs_stockloop18'

    
    config1 = {'value_nlayers':7,'value_hidden':100,'pol_nlayers':10,'pol_hidden':250,'pol_val_samedim':True,'dropout':0,'hidden_graph':256,'hidden_edge':128,'residual':True,'att_emb_size':515,'GAheads':5,'loss_func':'CE','lr': 0.001 ,'hidden': 4672, 'n_layers': 2,'heads': 16,'board_representation':'graph','useresnet':False}
    config2 = {'value_nlayers':5,'value_hidden':256,'pol_nlayers':5,'pol_hidden':256,'pol_val_samedim':True,'dropout':0,'hidden_graph':256,'hidden_edge':512,'residual':True,'att_emb_size':256,'GAheads':32,'loss_func':None,'lr': 0.0001, 'hidden': 2048, 'n_layers': 3,'heads': 32,'board_representation':'graph','useresnet':False}



    params1 = 'new_networks/new_graph_1k100k_params'
    params2 = 'new_networks/graph_1k100k_params'

    params3 = 'new_networks/new_graph_3k200k_params'
    params4 = 'new_networks/graph_3k200k_params'

    params5 = 'new_networks/new_graph_4k300k_params'
    params6 = 'new_networks/graph_4k300k_params'

    params7 = 'new_networks/new_graph_6k400k_params'
    params8 = 'new_networks/graph_6k400k_params'

    params9 = 'new_networks/new_graph_7k500k_params'
    params10 = 'new_networks/graph_7k500k_params'

    params11 = 'new_networks/new_graph_9k600k_params'
    params12 = 'new_networks/graph_9k600k_params'

    params13 = 'new_networks/new_graph_10k700k_params'
    params14 = 'new_networks/graph_10k700k_params'

    params15 = 'new_networks/new_graph_12k800k_params'
    params16 = 'new_networks/graph_12k800k_params'

    params17 = 'new_networks/new_graph_13k900k_params'
    params18 = 'new_networks/graph_13k900k_params'

    params19 = 'new_networks/new_graph_15k1000k_params'
    params20 = 'new_networks/graph_15k1000k_params'

    params21 = 'new_networks/new_graph_16k1100k_params'
    params22 = 'new_networks/graph_16k1100k_params'

    params = [params1,params2,params3,params4,params5,params6,params7,params8,params9,params10,params11,params12,params13,params14,params15,params16,params17,params18,params19,params20,params21,params22]
    # configs = [config1 for _ in range(len(params))]
    configs = [config1,config2,config1,config2,config1,config2,config1,config2,config1,config2,config1,config2,config1,config2,config1,config2,config1,config2,config1,config2,config1,config2]
    test_human(configs,params,10000)
    