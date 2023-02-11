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
import sys
import torch 
import pickle

np.set_printoptions(threshold=sys.maxsize)
class Node:
    def __init__(self,board, parent, prior,move):
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
        self.move = move

    def ave_eval(self):
        if len(self.evals) == 0:
            return 0
        return np.mean(self.evals)
    def ucb(self, c):
        p = c * self.P * np.sqrt(self.parent.n_visits) / (self.n_visits + 1)
        return -1*(self.ave_eval()) + p

    def expand(self,child_priors):
        # get policy and value from network
        if self.parent is None:
            for move in self.board.legal_moves:
                move_idx = encode_action(self.board,move)
                board_copy = copy.deepcopy(self.board)
                board_copy.push(move)
                child = Node(board_copy, self,child_priors[move_idx[0],move_idx[1],move_idx[2]],move=move)
                child.idx = move_idx
                self.children.append(child)  
            child_priors = add_noise(child_priors, self)
            for child in self.children:
                child.idx = move_idx
                child.P = child_priors[move_idx[0],move_idx[1],move_idx[2]]
        else:
            for move in self.board.legal_moves:
                move_idx = encode_action(self.board,move)
                board_copy = copy.deepcopy(self.board)
                board_copy.push(move)
                child = Node(board_copy, self,child_priors[move_idx[0],move_idx[1],move_idx[2]],move=move)
                child.idx = move_idx
                self.children.append(child)                            
        self.is_expanded = True

def select(node,c):
    while node.is_expanded:
        # get UCBs
        ucbs = np.array([i.ucb(c) for i in node.children])
        node = node.children[np.argmax(ucbs)]
    return node

def decode_outcome(outcome):
    if outcome.winner is None:
        return 0
    elif outcome.winner == chess.WHITE:
        return 1
    elif outcome.winner == chess.BLACK:
        return -1
    else:
        raise ValueError("Invalid outcome")
    
def add_noise(policy, node):
    new_policy = np.zeros((8,8,73))
    num_children = 0
    for child in node.children:
        num_children += 1
        new_policy[child.idx[0],child.idx[1],child.idx[2]] = policy[child.idx[0],child.idx[1],child.idx[2]]
    noise = np.random.dirichlet(np.zeros([num_children], dtype=np.float32)+0.3)
    for idx,child in enumerate(node.children):
        new_policy[child.idx[0],child.idx[1],child.idx[2]] += noise[idx]
    return new_policy

def backpropagate(node,result):
    node.n_visits += 1
    node.evals.append(result*node.turn)
    cur = node
    while cur.parent is not None:
        cur.n_visits += 1
        if cur.turn == node.turn:
            cur.evals.append(result)
        else:
            cur.evals.append(-1*result)

def mcts_run(root_state,net,c,num_runs):
    net.eval()
    root_node = Node(root_state,parent=None,prior=1,move=None)
    for i in range(num_runs):
        selected_node = select(root_node,c)
        value, policy = net([selected_node.graph])
        value = value[0].detach().numpy()[0]
        policy = policy[0].detach().numpy()
        if selected_node.board.outcome() is None:
            selected_node.expand(child_priors=policy)
        backpropagate(selected_node,value)
    net.train()
    # print(root_node.children[np.argmax([i.n_visits for i in root_node.children])].move)
    return root_node, root_node.children[np.argmax([i.n_visits for i in root_node.children])].board # return best move and root

def get_policy(node):
    policy = np.zeros([8,8,73])
    sum_visits = sum([i.n_visits for i in node.children])
    for child in node.children:
        policy[child.idx[0],child.idx[1],child.idx[2]] = child.n_visits/sum_visits
    return policy

def MCTS_selfplay(net,num_games=5000, num_sims_per_move=1600, train_freq = 100,buffer_size=100000, sample_size=10000,save_freq=500, eval_freq=200, calc_elo_freq=100):
    # initialize root node
    buffer = Buffer(max_size=buffer_size)
    for game in range(1,num_games):
        print('Game: ',game)
        polcies = []
        fens = []
        turns = []
        c = 2 if game < 100 else 0.7 # start with high exploration
        cur_board = chess.Board()
        count = 0
        value = 0
        pbar = tqdm(total=200)
        while cur_board.outcome() is None and count < 200:
            root, best_move_board = mcts_run(root_state=cur_board,net=net,c=c,num_runs=num_sims_per_move)
            turn = 1 if cur_board.turn == chess.WHITE else -1
            turns.append(turn)
            policy = get_policy(root)
            polcies.append(policy)
            fens.append(cur_board.fen())
            cur_board = best_move_board
            if cur_board.outcome() is not None:
                value = decode_outcome(cur_board.outcome())
            count += 1
            pbar.update(1)
            values = [value*turn for i in turns]
        pbar.close()
        assert len(fens) == len(polcies) == len(values)
        buffer.push(fens,values,polcies)
        if game % train_freq == 0:
            # train network on random batch of data
            data = buffer.sample(sample_size) 
            dataloader = DataLoader(data, batch_size=32, shuffle=True)
            trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=5)
            trainer.fit(net, dataloader)
    return root, net

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
torch.save(net, 'final_net')