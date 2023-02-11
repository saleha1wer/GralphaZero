from mcts import *
import pickle
import torch
"""
1. Self-Play for n games (MCTS)
    - Store all UCB move-selections for past 20*n games in buffer
2. Train network on a random batch of the past 20*n games for 1 epoch
3. Repeat 
- Every 100 loops, calc elo of network
- Every 500 loops, Evaluate against previous network (for 400 games) and replace best network if wins 55% of games
"""

def train():
    net = GNN({'lr': 0.1, 'hidden': 4672, 'n_layers': 1, 'batch_size': 32})
    root, net = MCTS_selfplay(net, 
                            num_games=10,
                            num_sims_per_move=777, 
                            train_freq = 5, 
                            buffer_size = 1000,
                            sample_size = 350,
                            save_freq=500, 
                            eval_freq=200, 
                            calc_elo_freq=100)

    with open('root.pkl', 'wb') as outp:
        pickle.dump(root, outp, pickle.HIGHEST_PROTOCOL)
    torch.save(net, 'final_net.pt')

if __name__ == '__main__':
    train()