import numpy as np
import multiprocessing as mp
import sys 
sys.path.append('.')
from utils.action_encoding import encode_action, decode_action,get_num_edges
from utils.board2graph import board2graph
from buffer import Buffer, join_buffers
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as torchDataLoader
import torch
from network import Network
import time 
from play import get_move
from load_data import get_stockfish_value_moves
import copy
from tqdm import tqdm
# from stockfish import Stockfish
import chess
import chess.engine
from scipy.special import softmax

def colect_stockfish_data(net,num_games,from_mcts=False,exploration=True,num_runs=400,depth=10,time=None):
    """
    - PLay N games
    - at each position, save:
        - board
        - stockfish value
        - stockfish policy
        - network policy
        - network value
        - if network turn, take action from network policy, else take action from stockfish policy (with exploration)
    - save positions in a buffer
    - return buffer
    """
    net.eval()
    boards, values, policies = [],[],[]
    # start stockfish
    # stockfish = Stockfish(r'/opt/homebrew/opt/stockfish/bin/stockfish',depth=depth)
    stockfish =  chess.engine.SimpleEngine.popen_uci(r'/opt/homebrew/opt/stockfish/bin/stockfish')
    limit = chess.engine.Limit(time=time) if time is not None else chess.engine.Limit(depth=depth)
    for n in tqdm(range(num_games)):
        board = chess.Board()
        white_list = [chess.WHITE if i % 2 == 0 else chess.BLACK for i in range(num_games)]
        # stockfish.set_fen_position(board.fen())
        while board.outcome() is None:
            # get evaluation and move list from stockfish
            legal_moves = list(board.legal_moves)
            turn_mult = 1 if board.turn == chess.WHITE else -1
            # moves = stockfish.get_top_moves(len(legal_moves))
            # eval = moves[0]['Centipawn'] if moves[0]['Mate'] is None else moves[0]['Mate']
            # eval = min(eval/700,1) if eval > 0 else max(eval/700,-1)
            # true_value = eval if board.turn == chess.WHITE else -1*eval
            # moves = {dict['Move']: [dict['Centipawn'],dict['Mate']] for dict in moves}
            # policy = np.array([moves[str(move)][0]*turn_mult if moves[str(move)][1] is None else (1/moves[str(move)][1]) *turn_mult*7500 for move in legal_moves])/25
            # policy = softmax(policy)
            result = stockfish.analyse(board, limit)
            eval = result['score'].white().score(mate_score=10000)
            eval = eval/650
            eval = min(eval,1) if eval > 0 else max(eval,-1)
            true_value = eval*turn_mult
            moves = result['pv']
            moves = [move for move in moves if move in legal_moves]
            probs = [len(moves)-i for i in range(len(moves))]
            probs = softmax(probs)
            idxs = [encode_action(board,move) for move in moves]
            policy = np.zeros(get_num_edges(board))
            policy[idxs] = probs 
            policies.append(policy)
            values.append(true_value)
            boards.append(copy.deepcopy(board))
            if white_list[n]==board.turn:
                # get pred_value and pred_policy
                _, pred_policy = net([board2graph(board)])
                pred_policy = pred_policy[0].detach().cpu().numpy().flatten()
                move = decode_action(board,pred_policy,exploration=exploration) 
            else:
                move = np.random.choice(moves)
            board.push(move)
            # stockfish.make_moves_from_current_position([move])
    assert len(boards) == len(values) == len(policies)
    buf = Buffer(max_size=np.inf,inc_preds=False)
    buf.push(boards,values,policies)
    stockfish.quit()
    return buf
 
def train_vs_stockfish(net_config,net_params,num_games,nloops,save_freq=5,from_mcts=False,num_runs=None,depth=15,time=None):
    network = Network(net_config)
    network.load_state_dict(torch.load(net_params))

    for n in range(7,nloops):
        save_name = net_params[:-1]+str(n)
        # save_name = net_params+'_vs_stock'+ 'loop{}'.format(n)
        print('save_name: ',save_name)
        buffer = colect_stockfish_data(net=network,num_games=num_games,from_mcts=from_mcts,num_runs=num_runs,depth=depth,time=time)
        network.train()
        data = buffer.sample_all()
        dataloader = DataLoader(data, batch_size=256, shuffle=True)
        trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1)
        trainer.fit(network, dataloader)
        if n % save_freq == 0:
            torch.save(network.state_dict(),save_name)
    return network

if __name__ == '__main__':

    config = {'loss_func':'ppo','lr': 0.0002 , 'hidden': 4672, 'n_layers': 2,'heads': 16,'gnn_type':'GAT','board_representation':'graph','useresnet':False}
    params = 'new_networks/new_graph_60k4000k_params_vs_stockloop6'

    train_vs_stockfish(config,params,num_games=500,nloops=25,save_freq=2,from_mcts=False,num_runs=None,depth=None,time=0.1)

    # network = train_vs_stockfish(config,params_path,50,100,10,False,self_play=True)
    # save_name =  'networks/{}_vsstock_final'.format(board_rep) if params_path is None else params_path+'_vsstock_final'
    # torch.save(network.state_dict(), save_name)