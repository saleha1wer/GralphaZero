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
from tqdm import tqdm
import pandas as pd
import pytorch_lightning as pl
import time
import math

class Node:
    def __init__(self,board, parent, prior):
        self.board = board
        self.graph = board2graph(board)
        self.turn = 1 if board.turn else -1
        self.parent = parent
        self.children = []
        self.n_visits = 1
        self.evals = []
        self.P = prior
        self.idx = None
        self.is_expanded = False

    def ave_eval(self):
        if len(self.evals) == 0:
            return 0
        return np.mean(self.evals)
    def ucb(self, c):
        p =  np.sqrt(self.parent.n_visits) / (self.n_visits + 1)
        return -1*(self.ave_eval()) + self.P + p*c

    def expand(self,net):
        # get policy and value from network
        _, policy = net([self.graph])
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

def select(node,net,c):
    fens = []
    policies = []
    turns = []
    # node.expand(net)
    while node.is_expanded:
        # update priors 
        _,policy = net([node.graph])
        policy = policy[0].detach().numpy()
        update_child_priors(node,policy)
        # get UCBs
        ucbs = np.array([i.ucb(c) for i in node.children])
        norm_ucbs = normalize([ucbs+np.abs(np.min(ucbs))])[0]
        #gather data
        fens.append(node.board.fen())
        turns.append(node.turn)
        policy = np.zeros((8,8,73))
        for i,child in enumerate(node.children):
            policy[child.idx[0],child.idx[1],child.idx[2]] = norm_ucbs[i]
        policies.append(policy)
        node = node.children[np.argmax(ucbs)]
    return node, fens,policies,turns

def decode_outcome(outcome):
    if outcome.winner is None:
        return 0
    elif outcome.winner == chess.WHITE:
        return 1
    elif outcome.winner == chess.BLACK:
        return -1
    else:
        raise ValueError("Invalid outcome")
    
def add_noise(policy):
    noise = np.random.dirichlet(np.zeros([4672], dtype=np.float32)+0.3)
    noise = noise.reshape((8,8,73))
    res = 0.75*policy + 0.25*noise
    return res

def simulate(node,net):
    # simulate game using network
    game = copy.deepcopy(node.board)
    while game.outcome() is None:
        _, policy = net([board2graph(game)])
        policy = policy[0].detach().numpy()
        policy = add_noise(policy)
        move = decode_action(game,policy)
        game.push(move)
    result = game.outcome()
    return decode_outcome(result)

def backpropagate(node,result):
    node.n_visits += 1
    node.evals.append(result*node.turn)
    if node.parent is not None:
        backpropagate(node.parent,result)

def mcts_run(root,net,buffer,c):
    # select node (save UCB distribution and turn) 
    leaf,fens,policies,turns = select(root,net,c)
    if leaf.board.outcome() is None:
        # expand node
        leaf.expand(net)
    # simulate game (using network)
    result = simulate(leaf,net)
    # backpropagate result
    backpropagate(leaf,result)
    # add data to buffer
    buffer.push(fens, list(np.array(turns)*result), policies)
    return root,buffer

def MCTS_selfplay(net,num_eps=5000, sims_per_ep=500, save_freq=500, eval_freq=200, calc_elo_freq=100):
    # initialize root node
    board = chess.Board()
    root = Node(board,parent=None,prior=1)
    buffer = Buffer(max_size=20*sims_per_ep)
    for ep in range(num_eps):
        print('Episode: ',ep)
        c = 2 if ep < 100 else 0.7 # start with high exploration
        for sim in tqdm(range(sims_per_ep)):
            root, buffer = mcts_run(root,net,buffer,c) 
        # train network on random batch 
        print('Buffer size: ',len(buffer.buffer))
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None,'display.max_colwidth', None):  # more options can be specified also
        #     for index, row in buffer.buffer.iterrows():
        #         print(row['fen'])
        #         print(row['value'])
        #         print(decode_action(chess.Board(row['fen']), row['policy']))
        data = buffer.sample(2000)
        dataloader = DataLoader(data, batch_size=32, shuffle=True)
        trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=3)
        trainer.fit(net, dataloader)
    #return net and root

net = GNN({'lr': 0.001, 'hidden': 4672, 'n_layers': 8, 'batch_size': 32})
MCTS_selfplay(net)
