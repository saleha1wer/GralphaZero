"""
Use network to play chess against opponent. 
Runs 1600 simulations of MCTS and selects the move with highest N value.
"""

import chess
from mcts import decode_outcome

def get_move(net, board):
    # Runs 1600 simulations of MCTS starting from board and selects the move with highest N value.
    pass

def play(net, opponent, white=True):
    # Plays a game of chess between net and opponent.
    # opponent is a function that takes in a board and returns a move.
    # white is a boolean that determines whether net plays white or black.
    game = chess.Board()
    while not game.is_game_over():
        if game.turn == white:
            move = get_move(net, game)
        else:
            move = opponent(game)
        game.push(move)
    return decode_outcome(game.outcome()) # 1 if white wins, -1 if black wins, 0 if draw
