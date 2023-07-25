"""
Use network to play chess against opponent. 
Runs 1600 simulations of MCTS and selects the move with highest N value.
"""
import chess
import chess.engine
from mcts import decode_outcome, mcts_run, get_policy
import numpy as np
from utils.action_encoding import decode_action,encode_action, old_decode_action
from utils.board2graph import board2graph
from utils.board2array import board2array
import torch
from stockfish import Stockfish
import chess.pgn as pgn
import warnings
from scipy.special import softmax
from datamodule import ChessDataset, ChessDataset_arrays
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as torchDataLoader

def get_move(net, board, c=0.7, num_runs=1600,from_mcts=True,exploration=False):
    # Runs 1600 simulations of MCTS starting from board and selects the move with highest N value.
    decode = decode_action if net.policy_format == 'graph' else old_decode_action
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if from_mcts:
        n_moves = len(list(board.legal_moves))
        num_runs = num_runs if n_moves > 10 else int(num_runs/4)
        num_runs = num_runs if n_moves > 5 else 35 
        num_runs = num_runs if n_moves > 1 else 2
        root, _ = mcts_run(root_state=board,net=net,c=c,num_runs=num_runs,disable_bar=False,exploration=exploration,board_rep=net.board_rep,skip_if_half=True)
        policy = get_policy(root,one_d_policy=net.policy_format=='graph')
        return decode(board,policy,exploration=exploration),root
    else:
        if net.board_rep == 'graph':
            single_dataset = ChessDataset([board], [1], [np.array([np.random.uniform() for _ in board.legal_moves])],log=False)
            single_loader = DataLoader(single_dataset, batch_size=1)
            for data in single_loader:
                data[0] = data[0].to(device)
                policy = net(data)[1][0]
        else:
            single_dataset = ChessDataset_arrays([board], [1], [torch.from_numpy(np.random.uniform(0,1,(8,8,73)))])
            single_loader = torchDataLoader(single_dataset, batch_size=1)
            for data in single_loader:
                data[0] = data[0].to(device)
                policy = net(data)[1]
        policy = policy.reshape(8, 8, 73) if net.policy_format == 'array' else policy
        policy = policy.detach().cpu().numpy()
        return decode(board, policy, exploration=exploration), policy

def _print_pgn(board,name1,name2):
    game_pgn = pgn.Game().from_board(board)
    game_pgn.headers['White'] = name1
    game_pgn.headers['Black'] = name2
    exporter = pgn.StringExporter(headers=True, variations=True, comments=True)
    print(game_pgn.accept(exporter))

def get_gui_info_mcts(root):
    pass
def play(net, opponent_rating,engine='stockfish',white=True, num_runs=1600, c=0.7,return_pgn=False,from_mcts=True,exploration=False,show_pgn=True, depth=None,level=None,gui=False,time_limit=None):
    # Plays a game of chess between net and opponent.
    # opponent_rating is the rating required of the engine used as opponent.
    # white is a boolean that determines whether net plays white or black.
    gui_info = {'stats':[], 'from_mcts':from_mcts}
    eng_path = r'/opt/homebrew/opt/stockfish/bin/stockfish' if engine == 'stockfish' else r'/opt/homebrew/opt/gnu-chess/bin/gnuchessu'
    if net.training:
        warnings.warn("Network is in training mode. Use net.eval() to switch to evaluation mode.")
        net.eval()
    game = chess.Board('rnb1k1nr/pppppppp/3q4/1N6/1b5N/1P2P3/PBPPBPPP/R3KR2 b Qkq - 0 1')
    if engine == 'stockfish':
        if time_limit is not None:
            depth = depth if depth is not None else 15
            engine = Stockfish(eng_path,depth=depth)
            engine.set_elo_rating(opponent_rating)
            engine._set_option("Minimum Thinking Time",1)
            if level is not None:
                engine.set_skill_level(level)
        else:
            engine = Stockfish(eng_path)
    
        engine.set_fen_position('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3 27')
    else:
        engine = chess.engine.SimpleEngine.popen_uci(eng_path)

    while game.outcome() is None:
        if game.turn == white:
            move,root_or_policy = get_move(net, game, c=c, num_runs=num_runs,from_mcts=from_mcts,exploration=exploration)
            if from_mcts:
                gui_info['stats'].append(get_gui_info_mcts(root_or_policy))
            else:
                gui_info['stats'].append({'move':str(move),'value':softmax(root_or_policy)[list(game.legal_moves).index(move)]})
        else:
            if engine == 'stockfish':
                if time_limit is not None:
                    move = engine.get_best_move()
                else:
                    move = engine.get_best_move_time(int(time_limit*1000))
                engine.make_moves_from_current_position([str(move)])
                move = chess.Move.from_uci(move)
            else:
                res = engine.play(game, chess.engine.Limit(time=time_limit))
                move = res.move
        game.push(move)
        if show_pgn:
            _print_pgn(game, 'Network' if white else 'Stockfish '+str(opponent_rating),'Stockfish '+str(opponent_rating) if white else 'Network')
    if engine != 'stockfish':
        engine.quit()
    if return_pgn:
        game_pgn = pgn.Game().from_board(game)
        game_pgn.headers['White'] = 'Network' if white else 'Stockfish '+str(opponent_rating)
        game_pgn.headers['Black'] = 'Stockfish '+str(opponent_rating) if white else 'Network'
        if gui:
            return decode_outcome(game.outcome()),game_pgn,gui_info
        else:
            return decode_outcome(game.outcome()),game_pgn
    return decode_outcome(game.outcome()) # 1 if white wins, -1 if black wins, 0 if draw

def rand_opp(board):
    # Random opponent, for debugging
    return np.random.choice(list(board.legal_moves))

# net = torch.load('network_human_games')
# print(play(net, 300, white=True, num_runs=777, c=0.7))

# stockfish_engine(300,[])