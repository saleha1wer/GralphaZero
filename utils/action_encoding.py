import numpy as np 

def encode_action(action):
    """
    Encodes an action as a matrix of size 8x8x73 following alphazero output format
    https://github.com/geochri/AlphaZero_Chess/blob/master/src/encoder_decoder.py
    """
    return np.random.uniform(size=(8,8,73)) 
    # perhaps just return the indicies of the action


def decode_action(encoded_action):
    """
    Decodes an action from a matrix of size 8x8x73 to a Chess.Move object
    https://github.com/geochri/AlphaZero_Chess/blob/master/src/encoder_decoder.py
    """
    return np.random.uniform(size=(8,8,73))
