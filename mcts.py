"""
MCTS self-play implementation
"""
import torch 
from buffer import Buffer
import chess
from utils.board2graph import board2graph
import copy
from utils.action_encoding import decode_action,encode_action
import numpy as np 
from tqdm import tqdm
import pytorch_lightning as pl
import time
import sys
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
        moves = list(self.board.legal_moves)
        for move in moves:
            move_idx = encode_action(self.board,move)
            board_copy = copy.deepcopy(self.board)
            board_copy.push(move)
            child = Node(board_copy, self,child_priors[move_idx[0],move_idx[1],move_idx[2]],move=move)
            child.idx = move_idx
            self.children.append(child) 
        if self.parent is None:
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
    cur = node
    while cur.parent is not None:
        cur.n_visits += 1
        result = -1*result if cur.turn != chess.WHITE else result
        cur.evals.append(result)
        cur = cur.parent
        if cur.parent is None:
            cur.n_visits += 1

def mcts_run(root_state,net,c,num_runs,disable_bar=True):
    # net.eval()
    root_node = Node(root_state,parent=None,prior=1,move=None)
    for i in tqdm(range(num_runs), disable=disable_bar):
        selected_node = select(root_node,c)
        value, policy = net([selected_node.graph])
        value = value[0].detach().numpy()[0]
        policy = policy[0].detach().numpy()
        if i == 0:
            print('value: ',value)
        if selected_node.board.outcome() is None:
            selected_node.expand(child_priors=policy)
        else:
            value = decode_outcome(selected_node.board.outcome())
        backpropagate(selected_node,value)
    # net.train()
    # print(root_node.children[np.argmax([i.n_visits for i in root_node.children])].move)
    return root_node, root_node.children[np.argmax([i.n_visits for i in root_node.children])].board # return best move and root

def get_policy(node):
    policy = np.zeros([8,8,73])
    sum_visits = sum([i.n_visits for i in node.children])
    # print('sum_visits: ',sum_visits)
    for child in node.children:
        policy[child.idx[0],child.idx[1],child.idx[2]] = child.n_visits/sum_visits
        # print('move:', child.move)
        # print('visits: ',child.n_visits)
        # print('prior: ', child.P)
        # print('ave eval: ', child.ave_eval())
    return policy

async def analyze_boards(boards_list):
    transport, engine = await chess.engine.popen_uci(r'/opt/homebrew/opt/stockfish/bin/stockfish')
    scores = []
    best_moves = []
    for b in boards_list:
        info = await engine.analyse(b, chess.engine.Limit(time=0.1))
        score = info['score'].white().score(mate_score=10000)
        moves = info['pv']
        move = moves[0] if len(moves) < 2 else moves[:2]
        best_moves.append(move)
        if score in [10000,-10000]:
            score = 1 if score > 0 else -1
        else:
            adjusted_score = score/1000
            score = min(adjusted_score,1) if score > 0 else max(adjusted_score,-1)
        scores.append(score)
    await engine.quit()
    return scores,best_moves

def get_stockfish_values_policies(boards):
    import asyncio
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    nscores,nbest_moves = asyncio.run(analyze_boards(boards))
    new_policies = []
    for idx,board in enumerate(boards):                       
        npolicy = np.zeros((8,8,73))
        move = nbest_moves[idx]
        if move.__class__ == list:
            act1_idx = encode_action(board,move[0])
            act2_idx = encode_action(board,move[1])
            act1_pol = np.random.uniform(0.5,0.8)
            npolicy[act1_idx[0],act1_idx[1],act1_idx[2]] =act1_pol
            npolicy[act2_idx[0],act2_idx[1],act2_idx[2]] = 1-act1_pol
        else:
            act_idx = encode_action(board,move)
            npolicy[act_idx[0],act_idx[1],act_idx[2]] = 1
        new_policies.append(npolicy)
    return nscores,new_policies

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
            root, best_move_board = mcts_run(root_state=cur_board,net=net,c=c,num_runs=num_sims_per_move,disable_bar=disable_mcts_bar)
            turn = 1 if cur_board.turn == chess.WHITE else -1
            turns.append(turn)
            policies.append(get_policy(root))
            boards.append(copy.deepcopy(cur_board))
            cur_board = best_move_board
            if cur_board.outcome() is not None:
                value = decode_outcome(cur_board.outcome())
            count += 1
            pbar.update(1)
        if stockfish:
            values,policies = get_stockfish_values_policies(boards) 
        else: 
            values,policies = [value for i in turns],policies
        pbar.close()
        print(values)
        for i in range(10):
            print(boards[i].fen())
        assert len(boards) == len(policies) == len(values)
        buffer.push(boards,values,policies)
    return buffer

# Testing
# net = GNN({'lr': 0.1, 'hidden': 4672, 'n_layers': 1, 'batch_size': 32})
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