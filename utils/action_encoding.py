import numpy as np 
import chess

"""
https://stats.stackexchange.com/questions/355994/representation-input-and-output-nodes-in-neural-network-for-textitalphazero
AlphaZero action encoding ^
"""
def encode_action(board,action):
    """
    Encodes an action as a matrix of size 8x8x73 following alphazero output format
    board: chess.Board object
    action: string (e.g. 'e2e4')
    https://github.com/geochri/AlphaZero_Chess/blob/master/src/encoder_decoder.py
    """
    return [np.random.randint(0,8),np.random.randint(0,8),np.random.randint(0,73)]
    # perhaps just return the indicies of the action


def decode_action(board, encoded_action):
    """
    Decodes an action from a matrix of size 8x8x73 to a Chess.Move object
    board: chess.Board object
    encoded_action: np.array of size 8x8x73
    https://github.com/geochri/AlphaZero_Chess/blob/master/src/encoder_decoder.py
    """
    return chess.Move(chess.E2,chess.E4) 
