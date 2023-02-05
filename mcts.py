"""
MCTS self-play implementation
"""
from network import GNN
from buffer import Buffer

class Node:
    def __init__(self,board, parent, prior):
        self.board = board
        self.parent = parent
        self.children = []
        self.n_visits = 0.1
        self.evals = [1]
        self.P = prior
    

def mcts_run(root,net,buffer):
    pass
    # initialize game
    # select node (save UCB distribution aka policy) 
    # expand node
    # simulate game (using network)
    # backpropagate result 
    # add data to buffer
    # return root, buffer

def MCTS_selfplay(net,num_eps=100, sims_per_ep=2500):
    for ep in range(num_eps):
        for sim in range(sims_per_ep):
            root, buffer = mcts_run(root,net,buffer)
            pass
        
        # train network on random batch of past 20*sims_per_ep games





net = GNN({'lr': 0.001, 'hidden': 4672, 'n_layers': 8, 'batch_size': 32})
MCTS_selfplay(net)

