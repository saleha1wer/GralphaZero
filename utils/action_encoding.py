import numpy as np 
import chess
import warnings
from scipy.special import softmax
from utils.board2graph import board2graph, encode_move_edge
from sklearn.preprocessing import normalize
"""
https://stats.stackexchange.com/questions/355994/representation-input-and-output-nodes-in-neural-network-for-textitalphazero
AlphaZero action encoding ^
"""
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

def square_str_to_filerank(square_name):
  assert len(square_name) == 2
  let = square_name[0]
  num = square_name[1]
  let = ord(let) - 97
  num = int(num) - 1
  return (7-num,let)

def square_indicies_to_str(square_idxs):
  num = 8 - square_idxs[0]
  let = chr(square_idxs[1]+97)
  sqr = str(let)+str(num)
  return sqr

def old_encode_action(board,action):
    """
    Encodes an action as a matrix of size 8x8x73 following alphazero output format
    return piece_rep, sqfrom_rep, sqto_rep (1,12), (8,8), (8,8,73)
    board: chess.Board object
    action: string (e.g. 'e2e4')
    https://github.com/geochri/AlphaZero_Chess/blob/master/src/encoder_decoder.py
    """
    i, j = square_str_to_filerank(chess.square_name(action.from_square))
    x ,y = square_str_to_filerank(chess.square_name(action.to_square))
    promoted = action.promotion
    dx, dy = x-i, y-j
    piece_type = board.piece_type_at(action.from_square)
    if piece_type != chess.KNIGHT and promoted in [None, chess.QUEEN]:
      if dx != 0 and dy == 0: # north-south idx 0-13
        idx = 7 + dx if dx < 0 else 6 + dx
      elif dx == 0 and dy != 0: # east-west idx 14-27
        idx = 21 + dy if dy < 0 else 20 + dy
      elif dx == dy: # NW-SE idx 28-41
        idx = 35 + dx if dx < 0 else 34 + dx
      elif dx == -dy: # NE-SW idx 42-55
        idx = 49 + dx if dx < 0 else 48 + dx
    elif piece_type == chess.KNIGHT:
      dic = {(i+2,j-1):56, (i+2,j+1):57,(i+1,j-2):58,(i-1,j-2):59,(i-2,j+1):60,(i-2,j-1):61,(i-1,j+2):62,(i+1,j+2):63}
      idx = dic[(x,y)]
    elif piece_type == chess.PAWN and (x == 0 or x == 7) and promoted != None:
      if abs(dx) == 1 and dy == 0:
        prom_to_idx = {chess.ROOK:64,chess.KNIGHT:65,chess.BISHOP:66}
        idx = prom_to_idx[promoted]
      if abs(dx) == 1 and dy == -1:
        prom_to_idx = {chess.ROOK:67,chess.KNIGHT:68,chess.BISHOP:69}
        idx = prom_to_idx[promoted]
      if abs(dx) == 1 and dy == 1:
        prom_to_idx = {chess.ROOK:70,chess.KNIGHT:71,chess.BISHOP:72}
        idx = prom_to_idx[promoted]
    # piece = board.piece_at(chess.square(j,7-i))
    # encoded_dict = {"R":0, "N":1, "B":2, "Q":3, "K":4, "P":5, "r":6, "n":7, "b":8, "q":9, "k":10, "p":11}
    return [i,j,idx]
    # return piece_rep, sqfrom_rep, sqto_rep (1,12), (8,8), (8,8,73)
  
def get_num_edges(board):
  _,graph = board2graph(board,return_nodes=True)
  selected_indices = ((graph.edge_attr[:, 0] == 1) & (graph.edge_attr[:, 1] == 0)).nonzero(as_tuple=True)[0]
  nodefrom = list(graph.edge_index[0][selected_indices])
  nodeto = list(graph.edge_index[1][selected_indices])
  assert len(nodefrom) == len(nodeto)
  num_edges = len(nodefrom)
  return num_edges
  
def encode_action(board,action):
    return list(board.legal_moves).index(action)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def decode_action(board, encoded_action,exploration=False):
  encoded_action = encoded_action.reshape(-1)
  # p = normalize_array(encoded_action)
  if True in np.isnan(encoded_action):
    print('NAN found')
    print(encoded_action)
    print(board.fen())
    encoded_action = np.nan_to_num(encoded_action)
    print(encoded_action)
  p = softmax(encoded_action)
  best_idx = np.random.choice(list(range(encoded_action.shape[0])), p=p) if exploration else np.argmax(encoded_action) 
  move = list(board.legal_moves)[best_idx]
  return move

def old_decode_action(board, encoded_action,exploration=False):
    """
    Decodes an action from a matrix of size 8x8x73 to a Chess.Move object
    board: chess.Board object
    encoded_action: np.array of size 8x8x73
    https://github.com/geochri/AlphaZero_Chess/blob/master/src/encoder_decoder.py
    """
    legal_moves_idx = np.array([np.array(old_encode_action(board,i)) for i in board.legal_moves])
    new_encoded_action = np.zeros((8,8,73))
    new_encoded_action.fill(-np.inf)
    new_encoded_action[legal_moves_idx[:,0],legal_moves_idx[:,1],legal_moves_idx[:,2]] = encoded_action[legal_moves_idx[:,0],legal_moves_idx[:,1],legal_moves_idx[:,2]]
    # new_encoded_action = new_encoded_action + np.abs(np.min(new_encoded_action))
    best_idx = legal_moves_idx[np.random.choice(range(legal_moves_idx.shape[0]),p=softmax(new_encoded_action[legal_moves_idx[:,0],legal_moves_idx[:,1],legal_moves_idx[:,2]]))] if exploration else np.unravel_index(new_encoded_action.argmax(), new_encoded_action.shape)
    if np.max(new_encoded_action) == 0:
      warnings.warn("Encoded action is ilegal, returning first legal move.")
    i,j = best_idx[0],best_idx[1]
    initial_pos = (i,j)
    k = best_idx[2]
    promoted = None
    if k <= 13:
      dy = 0
      dx = k - 7 if k < 7 else k - 6
      final_pos = (i + dx, j + dy)
    elif 14 <= k <= 27:
      dx = 0
      dy = k - 21 if k < 21 else k - 20
      final_pos = (i + dx, j + dy)
    elif 28 <= k <= 41:
      dy = k - 35 if k < 35 else k - 34
      dx = dy
      final_pos = (i + dx, j + dy)
    elif 42 <= k <= 55:
      dx = k - 49 if k < 49 else k - 48
      dy = -dx
      final_pos = (i + dx, j + dy)
    elif 56 <= k <= 63:
      dic = {56:(2,-1), 57:(2,1),58:(1,-2),59:(-1,-2),60:(-2,1),61:(-2,-1),62:(-1,2),63:(1,2)}
      dx = dic[k][0]
      dy = dic[k][1]
      final_pos = (i + dx, j + dy)
    elif 64 <= k <= 66:
      turn_to_pos = {chess.WHITE: (i-1,j), chess.BLACK: (i+1,j)}
      k_to_piece = {64: chess.ROOK, 65: chess.KNIGHT,66:chess.BISHOP}
      final_pos = turn_to_pos[board.turn]
      promoted = k_to_piece[k]
    elif 67 <= k <= 69:
      turn_to_pos = {chess.WHITE: (i-1,j-1), chess.BLACK: (i+1,j-1)}
      k_to_piece = {67: chess.ROOK, 68: chess.KNIGHT,69:chess.BISHOP}
      final_pos = turn_to_pos[board.turn]
      promoted = k_to_piece[k]
    elif 70 <= k <= 72:
      turn_to_pos = {chess.WHITE: (i-1,j+1), chess.BLACK: (i+1,j+1)}
      k_to_piece = {70: chess.ROOK, 71: chess.KNIGHT,72:chess.BISHOP}
      final_pos = turn_to_pos[board.turn]
      promoted = k_to_piece[k]

    init_sqr = chess.parse_square(square_indicies_to_str(initial_pos))
    final_sqr = chess.parse_square(square_indicies_to_str(final_pos))

    if board.piece_type_at(init_sqr) == chess.PAWN and final_pos[0] in [0,7] and promoted == None:
      promoted = chess.QUEEN
    move = chess.Move(init_sqr,final_sqr,promotion=promoted)
    return move


if __name__ == "__main__":
  board = chess.Board('6nr/p4pk1/1R6/2p2b1p/1p6/2P3P1/1P2rPBP/R1B3K1 b - - 1 23')
  for move in board.legal_moves:
    _,_,encoded_action = encode_action(board,move)
    enc = np.zeros((8,8,73))
    enc.fill(0.3)
    enc[encoded_action[0],encoded_action[1],encoded_action[2]] = 0.5
    dec_move = decode_action(board,enc)
    if move != dec_move:
      print('ERROR')
      raise Exception('Move not decoded correctly')
    else:
      print('OK')
      print('original move: ',move)
      print('encoded action: ',encoded_action)
      print('decoded move: ',dec_move)


