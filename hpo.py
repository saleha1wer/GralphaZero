"""
- Define Hyperparameter space (fixed params and dataloader here)
- Define objective function (load two buffers and train model, 2 epochs on each. return MSE and Cross Entropy on unseen data (same data for all models))
    - takes in hyperparameters and two dataloaders as input (dataloader for training and dataloader for validation)
    - returns MSE and Cross Entropy on unseen data
- Choose optimization algorithm
- Run optimization
"""
import time
from network import Network
import pytorch_lightning as pl
import torch
from bohb.configspace import ConfigurationSpace, CategoricalHyperparameter, UniformHyperparameter, IntegerUniformHyperparameter
from bohb import BOHB
import bohb.configspace as cs
from train_human_games.load_data import load_data
from torch_geometric.loader import DataLoader

def create_model(params):
    # This function will create a model with the given hyperparameters. 
    # You should replace it with your own model's constructor.
    config =  {'board_representation':'graph',
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
            'lr': 0.0001}
    model = Network(config)
    model.train()
    return model

def objective(train_dataloader, val_dataloader, params):
    # Create the model using the hyperparameters
    print('Params')
    print(params)
    model = create_model(params)
    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger=pl.loggers.TensorBoardLogger('HPO_logs/', name=f"experiment_{time.time()}")
    trainer = pl.Trainer(accelerator=device, devices=1, max_epochs=3)
    trainer.fit(model, train_dataloader)
    # Evaluate the model
    loss = trainer.test(model, val_dataloader)[0]
    # hparams = {**params, 'loss': loss}
    metrics={'cross_entropy':loss['cross_entropy'], 'best_move_acc':loss['best_move_acc'], 'mse':loss['mse']}
    logger.log_hyperparams(params, metrics=metrics)
    # trainer.logger.log_metrics(metrics)
    loss = (loss['cross_entropy']+loss['best_move_acc'])/2 + loss['mse']
    return loss

if __name__ == '__main__':
    # Load your data

    value_nlayers = CategoricalHyperparameter('value_nlayers', [1,5,7,15])
    pol_nlayers = CategoricalHyperparameter('pol_nlayers', [1,5,7,15])
    hidden_graph = CategoricalHyperparameter('hidden_graph', [256,512,1024,2048])
    hidden_edge = CategoricalHyperparameter('hidden_edge', [256,512,1024,2048])
    heads_GAT_edge = CategoricalHyperparameter('heads_GAT_edge', [16,32,64,128,256])
    heads_GAT_graph = CategoricalHyperparameter('heads_GAT_graph', [16,32,64,128,256])
    att_emb_size = CategoricalHyperparameter('att_emb_size', [128,256,512,1024,2048])
    GAheads = CategoricalHyperparameter('GAheads', [16,32,64,128])
    n_layers = CategoricalHyperparameter('n_layers', [1,3,5,7,9,11])
    config_space = ConfigurationSpace(hyperparameters=[hidden_graph, hidden_edge, att_emb_size, value_nlayers, pol_nlayers, heads_GAT_graph,heads_GAT_edge, n_layers, GAheads])
    train_range, train_n_rand_pos = (0,1000), 200000
    val_range, val_n_rand_pos = (1000,1250), 50000
    # train_range, train_n_rand_pos = (0,10), 20
    # val_range, val_n_rand_pos = (1000,1005), 10
    # Load the data
    train_buffer = load_data(path='train_human_games/human_games/games_chesstempo.csv',
            policy_format='graph',
            testing=True,
            stockfish_value=True,
            testing_range=train_range,
            save_path=None,
            time_limit=0.01,
            n_random_pos=train_n_rand_pos)
    train_data = train_buffer.sample_all(dataset_type='graph',policy_format='graph')
    train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True,drop_last=True)
    val_buffer = load_data(path='train_human_games/human_games/games_chesstempo.csv',
            policy_format='graph',
            testing=True,
            stockfish_value=True,
            testing_range=val_range,
            save_path=None,
            time_limit=0.01,
            n_random_pos=val_n_rand_pos)
    val_data = val_buffer.sample_all(dataset_type='graph',policy_format='graph')
    val_dataloader = DataLoader(val_data, batch_size=256, shuffle=True,drop_last=True)

    # Define the optimization function
    def evaluate(params, n_iterations):
        return objective(train_dataloader, val_dataloader, params)

    # Create and run the optimizer
    opt = BOHB(config_space, evaluate, max_budget=10, min_budget=1)
    logs = opt.optimize()
    print(logs)