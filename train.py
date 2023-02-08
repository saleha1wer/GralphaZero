from mcts import *
import pickle
"""
1. Self-Play for n games (MCTS)
    - Store all UCB move-selections for past 20*n games in buffer
2. Train network on a random batch of the past 20*n games for 1 epoch
3. Repeat 
- Every 100 loops, calc elo of network
- Every 500 loops, Evaluate against previous network (for 400 games) and replace best network if wins 55% of games
"""


def train():
    net = GNN({'lr': 0.001, 'hidden': 4672, 'n_layers': 8, 'batch_size': 32})
    root, net = MCTS_selfplay(net, num_eps=5000, sims_per_ep=500, save_freq=500, eval_freq=200, calc_elo_freq=100)
    with open('root.pkl', 'wb') as outp:
        pickle.dump(root, outp, pickle.HIGHEST_PROTOCOL)
    

    