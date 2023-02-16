"""
Use network to play chess against opponent. 
Runs 1600 simulations of MCTS and selects the move with highest N value.
"""
import chess
from mcts import decode_outcome, mcts_run, get_policy
import numpy as np
from utils.action_encoding import decode_action,encode_action
import torch
from stockfish import Stockfish
import chess.pgn as pgn


def stockfish_engine(rating,moves_list):
    stockfish = Stockfish()
    stockfish.set_elo_rating(rating)
    stockfish.set_fen_position('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    stockfish.make_moves_from_current_position(moves_list)
    return stockfish.get_best_move()

def get_move(net, board, c=0.7, num_runs=1600):
    # Runs 1600 simulations of MCTS starting from board and selects the move with highest N value.
    root, _ = mcts_run(root_state=board,net=net,c=c,num_runs=num_runs,disable_bar=True)
    return decode_action(board,get_policy(root))

def play(net, opponent_rating, white=True, num_runs=1600, c=0.7,return_pgn=False):
    # Plays a game of chess between net and opponent.
    # opponent_rating is the rating required of the engine used as opponent.
    # white is a boolean that determines whether net plays white or black.
    net.eval()
    game = chess.Board()
    moves_list = []
    while game.outcome() is None:
        if game.turn == white:
            move = get_move(net, game, c=c, num_runs=num_runs)
            moves_list.append(str(move))
        else:
            move = stockfish_engine(opponent_rating,moves_list)
            moves_list.append(str(move))
            move = chess.Move.from_uci(move)
        game.push(move)
    net.train()
    if return_pgn:
        game_pgn = pgn.Game().from_board(game)
        game_pgn.headers['White'] = 'Network' if white else 'Stockfish '+str(opponent_rating)
        game_pgn.headers['Black'] = 'Stockfish '+str(opponent_rating) if white else 'Network'
        return decode_outcome(game.outcome()),game_pgn
    return decode_outcome(game.outcome()) # 1 if white wins, -1 if black wins, 0 if draw

def rand_opp(board):
    # Random opponent, for debugging
    return np.random.choice(list(board.legal_moves))

# net = torch.load('final_net')
# print(play(net, 300, white=True, num_runs=100, c=0.7))

