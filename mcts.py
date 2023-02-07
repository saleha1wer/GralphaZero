"""
MCTS self-play implementation
"""
from network import GNN
from buffer import Buffer
import chess
from utils.board2graph import board2graph
from datamodule import ChessDataset
import copy
from torch_geometric.loader import DataLoader
from utils.action_encoding import decode_action,encode_action
import numpy as np 
from sklearn.preprocessing import normalize


class Node:
    def __init__(self,board, parent, prior):
        self.board = board
        self.graph = board2graph(board)
        self.turn = 1 if board.turn else -1
        self.parent = parent
        self.children = []
        self.n_visits = np.random.randint(20)
        self.evals = [np.random.choice([-1,1]) for i in range(self.n_visits)]
        self.P = prior
        self.idx = None
        self.is_expanded = False

    def ave_eval(self):
        if len(self.evals) == 0:
            return 0
        return np.mean(self.evals)
    def ucb(self):
        p = self.P * np.sqrt((np.log(self.parent.n_visits))/ (np.log(1 + self.n_visits))) 
        return self.ave_eval() + p

    def expand(self,net):
        # get policy and value from network
        value, policy = net([self.graph])
        policy = policy[0].detach().numpy()
        if self.is_expanded:
            # update priors
            update_child_priors(self,policy)
        else:
            # for each move, create child node
            for move in self.board.legal_moves:
                move_idx = encode_action(self.board,move)
                board_copy = copy.deepcopy(self.board)
                board_copy.push(move)
                child = Node(board_copy, self,policy[move_idx[0],move_idx[1],move_idx[2]])
                child.idx = move_idx
                self.children.append(child)                            
        self.is_expanded = True

def update_child_priors(node,policy):
    for child in node.children:
        child.P = policy[child.idx[0],child.idx[1],child.idx[2]]

def select(node,net):
    fens = []
    policies = []
    turns = []
    # node.expand(net)
    while node.is_expanded:
        # update priors 
        _,policy = net([node.graph])
        policy = policy[0].detach().numpy()
        update_child_priors(node,policy)
        ucbs = np.array([i.ucb() for i in node.children])
        norm_ucbs = normalize([ucbs+np.abs(np.min(ucbs))])[0]
        #gather data
        for i,child in enumerate(node.children):
            fens.append(child.board.fen())
            turns.append(child.turn)
            policy = np.zeros((8,8,73))
            policy[child.idx[0],child.idx[1],child.idx[2]] = norm_ucbs[i]
            policies.append(policy)
        node = node.children[np.argmax(ucbs)]
    return node, fens,policies,turns

def simulate(node):
    raise

def mcts_run(root,net,buffer):
    # select node (save UCB distribution and turn) 
    leaf,fens,policies,turns = select(root,net)
    # expand node
    leaf.expand(net)
    raise
    # simulate game (using network)
    # backpropagate result (and save value*turn)
    # add data to buffer
    buffer.push(...)
    return root,buffer

def MCTS_selfplay(net,num_eps=5000, sims_per_ep=500, save_freq=500, eval_freq=200, calc_elo_freq=100):
    # initialize root node
    board = chess.Board()
    root = Node(board,parent=None,prior=1)
    buffer = Buffer(max_size=20*sims_per_ep)
    for ep in range(num_eps):
        for sim in range(sims_per_ep):
            root, buffer = mcts_run(root,net,buffer) # start with high exploration and decrease over time
            pass

        # train network on random batch of past 20*sims_per_ep games


net = GNN({'lr': 0.001, 'hidden': 4672, 'n_layers': 8, 'batch_size': 32})
MCTS_selfplay(net)
# 