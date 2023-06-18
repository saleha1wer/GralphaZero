from utils.action_encoding import decode_action,encode_action
import chess
import numpy as np
import chess.engine

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
            adjusted_score = score/650
            score = min(adjusted_score,1) if score > 0 else max(adjusted_score,-1)
        score = score if b.turn == chess.WHITE else -1*score
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
            _,_,act1_idx = encode_action(board,move[0])
            _,_,act2_idx = encode_action(board,move[1])
            act1_pol = np.random.uniform(0.5,0.8)
            npolicy[act1_idx[0],act1_idx[1],act1_idx[2]] =act1_pol
            npolicy[act2_idx[0],act2_idx[1],act2_idx[2]] = 1-act1_pol
        else:
            act_idx = encode_action(board,move)
            npolicy[act_idx[0],act_idx[1],act_idx[2]] = 1
        new_policies.append(npolicy)
    return nscores,new_policies