"""
- Define Hyperparameter space (fixed params and dataloader here)
- Define objective function (load two buffers and train model, 2 epochs on each. return MSE and Cross Entropy on unseen data (same data for all models))
    - takes in hyperparameters and two dataloaders as input (dataloader for training and dataloader for validation)
    - returns MSE and Cross Entropy on unseen data
- Choose optimization algorithm
- Run optimization
"""
from hyperopt import fmin, tpe, hp,STATUS_OK
import hyperopt
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from network import Network
import time
from train_human_games.load_data import load_data
import gc

def create_model(params):
    config =  {
        'board_representation':'graph',
        'policy_format':'graph',
        'value_nlayers':params['value_nlayers'],
        'pol_nlayers':params['pol_nlayers'],
        'hidden_graph':params['hidden_graph'],
        'hidden_edge':params['hidden_edge'],
        'att_emb_size':params['att_emb_size'],
        'GAheads':params['GAheads'],
        'n_layers': params['n_layers'],
        'heads_GAT_graph': params['heads_GAT_graph'],
        'heads_GAT_edge': params['heads_GAT_edge'],
        'lr': 0.0001
    }
    model = Network(config)
    model.train()
    return model

if __name__ == '__main__':

    # Define the search space
    space = {
        'value_nlayers': hp.choice('value_nlayers', [1,5,7,15]),
        'pol_nlayers': hp.choice('pol_nlayers', [1,5,7,15]),
        'hidden_graph': hp.choice('hidden_graph', [256,512,1024,2048]),
        'hidden_edge': hp.choice('hidden_edge', [256,512,1024,2048]),
        'heads_GAT_edge': hp.choice('heads_GAT_edge', [16,32,64,128,256]),
        'heads_GAT_graph': hp.choice('heads_GAT_graph', [16,32,64,128,256]),
        'att_emb_size': hp.choice('att_emb_size', [128,256,512,1024,2048]),
        'GAheads': hp.choice('GAheads', [16,32,64,128]),
        'n_layers': hp.choice('n_layers', [1,3,5,7,9,11])
    }
    space = {
        'value_nlayers': hp.choice('value_nlayers', [1]),
        'pol_nlayers': hp.choice('pol_nlayers', [1]),
        'hidden_graph': hp.choice('hidden_graph', [32]),
        'hidden_edge': hp.choice('hidden_edge', [32]),
        'heads_GAT_edge': hp.choice('heads_GAT_edge', [2]),
        'heads_GAT_graph': hp.choice('heads_GAT_graph', [2,4]),
        'att_emb_size': hp.choice('att_emb_size', [48]),
        'GAheads': hp.choice('GAheads', [2]),
        'n_layers': hp.choice('n_layers', [1,1])
    }

    # Load the data
    # the same as your code above
    import asyncio
    # train_range, train_n_rand_pos = (0,1250), 200000
    # val_range, val_n_rand_pos = (1250,1500), 50000
    train_range, train_n_rand_pos = (0,10), 20
    val_range, val_n_rand_pos = (1000,1005), 10
    train_buffer =asyncio.run(load_data(path='train_human_games/human_games/games_chesstempo.csv',
            policy_format='graph',
            testing=True,
            stockfish_value=True,
            testing_range=train_range,
            save_path=None,
            time_limit=0.01,
            n_random_pos=train_n_rand_pos))
    train_data = train_buffer.sample_all(dataset_type='graph',policy_format='graph')
    train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True,drop_last=True)
    val_buffer = asyncio.run(load_data(path='train_human_games/human_games/games_chesstempo.csv',
            policy_format='graph',
            testing=True,
            stockfish_value=True,
            testing_range=val_range,
            save_path=None,
            time_limit=0.01,
            n_random_pos=val_n_rand_pos))
    val_data = val_buffer.sample_all(dataset_type='graph',policy_format='graph')
    val_dataloader = DataLoader(val_data, batch_size=256, shuffle=True,drop_last=True)
    def objective(params):
        print('Params')
        print(params)
        model = create_model(params)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger=pl.loggers.TensorBoardLogger('HPO_logs/', name=f"experiment_{time.time()}")
        trainer = pl.Trainer(accelerator=device, devices=1, max_epochs=3)
        model.to(device)
        trainer.fit(model, train_dataloader)
        loss = trainer.test(model, val_dataloader,verbose=False)[0]
        metrics={'cross_entropy':loss['cross_entropy'], 'best_move_acc':loss['best_move_acc'], 'mse':loss['mse']}
        logger.log_hyperparams(params, metrics=metrics)
        loss = loss['cross_entropy'] + loss['mse']
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return {'loss': loss, 'status': STATUS_OK}
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100)
    print(best)
    print(hyperopt.space_eval(space, best))
##rm hpo logs before starting##
##rm hpo logs before starting##
##rm hpo logs before starting##
##rm hpo logs before starting##
##rm hpo logs before starting##
##rm hpo logs before starting##
##rm hpo logs before starting##
##rm hpo logs before starting##
##rm hpo logs before starting##