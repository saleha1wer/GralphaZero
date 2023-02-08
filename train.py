"""
1. Self-Play for n games (MCTS)
    - Store all UCB move-selections for past 20*n games in buffer
2. Train network on a random batch of the past 20*n games for 1 epoch
3. Repeat 
- Every 100 loops, calc elo of network
- Every 1,000 loops, Evaluate against previous network (for 400 games) and save if wins 55% of games
- In alphazero, n = 25,000 and batch_size = 2048
"""

from mcts import *
import pickle


def train():
    net = GNN({'lr': 0.001, 'hidden': 4672, 'n_layers': 8, 'batch_size': 32})
    root, net = MCTS_selfplay(net, num_eps=5000, sims_per_ep=500, save_freq=500, eval_freq=200, calc_elo_freq=100)
    with open('root.pkl', 'wb') as outp:
        pickle.dump(root, outp, pickle.HIGHEST_PROTOCOL)
    

    