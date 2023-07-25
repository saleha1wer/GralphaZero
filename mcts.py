"""
MCTS self-play implementation
"""
import torch 
from buffer import Buffer
import chess
from utils.board2graph import board2graph
from utils.board2array import board2array
import copy
from utils.action_encoding import decode_action,encode_action, old_decode_action,old_encode_action
import numpy as np 
from tqdm import tqdm
import pytorch_lightning as pl
import time
import sys
np.set_printoptions(threshold=sys.maxsize)
from stock import get_stockfish_values_policies
from scipy.special import softmax
def normalize_array(arr, epsilon=1e-9):
    arr = np.array(arr)  # Convert the input to a numpy array if it isn't already
    if arr.size == 1:
        return np.array([1.0])
    min_value = np.min(arr)
    
    # Shift all elements by the absolute value of the smallest number
    shifted_arr = arr + abs(min_value)
    
    # Calculate the sum of the shifted array
    shifted_sum = np.sum(shifted_arr)
    
    # Add epsilon to the denominator to prevent division by zero
    normalized_arr = shifted_arr / (shifted_sum + epsilon)
    
    return normalized_arr

class Node:
    def __init__(self,board, parent, prior,move,board_rep):
        self.board = board
        if board_rep == 'graph':
            self.graph = board2graph(board)
        elif board_rep == 'array':
            self.graph = board2array(board)
        self.turn = 1 if board.turn else -1
        self.parent = parent
        self.children = []
        self.n_visits = 0
        self.evals = []
        self.value = 0
        self.P = prior
        self.idx = None
        self.is_expanded = False
        self.move = move

    def ave_eval(self):
        if len(self.evals) == 0:
            return 0
        return np.mean(self.evals)
    def ucb(self, c): 
        p = c *  self.P * np.sqrt(self.parent.n_visits) / (self.n_visits + 1)
        return -1*(self.ave_eval()) + p 

    def expand(self,child_priors,board_rep,noisy_root=True):
        # get policy and value from network
        moves = list(self.board.legal_moves)
        for move in moves:
            if len(child_priors.shape) > 2:
                _,_,move_idx = old_encode_action(self.board,move)
            else:
                move_idx = encode_action(self.board,move)
            board_copy = copy.deepcopy(self.board)
            board_copy.push(move)
            p = child_priors[move_idx[0],move_idx[1],move_idx[2]] if len(child_priors.shape) > 2 else child_priors[move_idx][0]
            child = Node(board_copy, self,p,move=move,board_rep=board_rep)
            child.idx = move_idx
            self.children.append(child) 
        if self.parent is None and noisy_root:
            child_priors = add_noise(child_priors, self)
            for child in self.children:
                m_idx = child.idx
                child.P = child_priors[m_idx[0],m_idx[1],m_idx[2]]
                          
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
    org_turn = node.turn
    cur = node
    while cur.parent is not None:
        cur.n_visits += 1
        result = result if cur.turn == org_turn else -1*result
        cur.evals.append(result)
        cur = cur.parent
        if cur.parent is None:
            cur.n_visits += 1

def get_mask(legal_moves_idx):
    mask = np.zeros((8,8,73))
    mask.fill(False)
    mask[legal_moves_idx[:,0],legal_moves_idx[:,1],legal_moves_idx[:,2]] = True
    return mask 

def legal_softmax(policy,board):
    legal_moves_idx = np.array([np.array(old_encode_action(board,i)[2]) for i in board.legal_moves])
    mask = get_mask(legal_moves_idx) 
    policy = policy * mask
    policy[legal_moves_idx[:,0],legal_moves_idx[:,1],legal_moves_idx[:,2]] = softmax(policy[legal_moves_idx[:,0],legal_moves_idx[:,1],legal_moves_idx[:,2]])
    return policy

def mcts_run(root_state,net,c,num_runs,board_rep,disable_bar=True,exploration=True,skip_if_half=False):
    # net.eval()
    root_node = Node(root_state,parent=None,prior=1,move=None,board_rep=board_rep)
    for i in tqdm(range(num_runs), disable=disable_bar):
        selected_node = select(root_node,c)
        print(selected_node.board.fen())
        t = net([selected_node.graph])
        value, policy = t[0], t[1]
        value = value[0].detach().cpu().numpy()[0]
        policy = policy[0].detach().cpu().numpy()
        selected_node.value = value
        if i == 0:
            print('value: ',value)
            print('fen: ', selected_node.board.fen())
        if selected_node.board.outcome() is None:
            policy = softmax(policy) if net.policy_format == 'graph' else legal_softmax(policy,selected_node.board)
            selected_node.expand(child_priors=policy,noisy_root=exploration,board_rep=board_rep)
        else:
            value = decode_outcome(selected_node.board.outcome())
            value = value * selected_node.turn
        backpropagate(selected_node,value)
        if skip_if_half:
            nvisits = [child.n_visits for child in root_node.children]
            if max(nvisits) > num_runs/2:
                break
    # net.train()
    # print(root_node.children[np.argmax([i.n_visits for i in root_node.children])].move)
    if exploration:
        best_move_probs = dict()
        for child in root_node.children:
            best_move_probs[child.move] = child.n_visits/root_node.n_visits
        return root_node, best_move_probs
    else:
        return root_node, root_node.children[np.argmax([i.n_visits for i in root_node.children])].board # return best move and root

def get_policy(node,one_d_policy=False):
    if one_d_policy:
        policy = np.zeros((len(list(node.board.legal_moves))))
    else:
        policy = np.zeros([8,8,73])
    sum_visits = node.n_visits
    vals = [child.n_visits/sum_visits for child in node.children]
    for idx,child in enumerate(node.children):
        if one_d_policy:
            policy[child.idx] = vals[idx]
        else:
            policy[child.idx[0],child.idx[1],child.idx[2]] = vals[idx]
        print('move:', child.move)
        print('visits: ',child.n_visits)
        print('prior: ', child.P)
        print('ave eval: ', child.ave_eval())
        print('value: ', child.value)
    return policy

def MCTS_selfplay(net,c,num_games=5000, num_sims_per_move=1600, buffer_size=None, disable_bar=False,disable_mcts_bar=True,stockfish=False):
    # initialize root node
    buffer_size = np.inf if buffer_size is None else buffer_size
    buffer = Buffer(max_size=buffer_size)
    for game in range(1,num_games+1):
        print('Game: ',game)
        policies, boards, turns = ([] for _ in range(3))
        cur_board = chess.Board()
        count, value = 0,0
        pbar = tqdm(total=200,disable=disable_bar)
        while cur_board.outcome() is None and count < 200:
            root, best_move_prob = mcts_run(root_state=cur_board,net=net,c=c,num_runs=num_sims_per_move,disable_bar=disable_mcts_bar,exploration=True)
            turn = 1 if cur_board.turn == chess.WHITE else -1
            turns.append(turn)
            policies.append(get_policy(root))
            boards.append(copy.deepcopy(cur_board))
            #TODO: softmax instead of visits/sum_visits
            best_move = np.random.choice(list(best_move_prob.keys()), p=list(best_move_prob.values()))
            cur_board.push(best_move)
            if cur_board.outcome() is not None:
                value = decode_outcome(cur_board.outcome())
            count += 1
            pbar.update(1)
        if stockfish:
            values,policies = get_stockfish_values_policies(boards) 
        else: 
            values,policies = [value * i for i in turns],policies
        pbar.close()
        assert len(boards) == len(policies) == len(values)
        buffer.push(boards,values,policies)
    return buffer

# Testing
# net = Network({'lr': 0.1, 'hidden': 4672, 'n_layers': 1, 'batch_size': 32})
# root, net = MCTS_selfplay(net, 
#                         num_games=10,
#                         num_sims_per_move=500, 
#                         train_freq = 2, 
#                         buffer_size = 500,
#                         sample_size = 350,
#                         save_freq=500, 
#                         eval_freq=200, 
#                         calc_elo_freq=100)

# torch.save(net, 'final_net')