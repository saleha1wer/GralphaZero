import numpy as np 

def encode_action(board,action):
    """
    Encodes an action as a matrix of size 8x8x73 following alphazero output format
    board: chess.Board object
    action: string (e.g. 'e2e4')
    https://github.com/geochri/AlphaZero_Chess/blob/master/src/encoder_decoder.py
    """
    return np.random.uniform(size=(8,8,73)) 
    # perhaps just return the indicies of the action


def decode_action(board, encoded_action):
    """
    Decodes an action from a matrix of size 8x8x73 to a Chess.Move object
    board: chess.Board object
    encoded_action: np.array of size 8x8x73
    https://github.com/geochri/AlphaZero_Chess/blob/master/src/encoder_decoder.py
    """
    return np.random.uniform(size=(8,8,73))
