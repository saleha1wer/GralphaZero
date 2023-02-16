from play import play
import torch 
import multiprocessing as mp
import chess.pgn as PGN

def update_elo(org_rating, opp_rating,score, k=32):
    """
    Updates the elo of the winner and loser
    """
    E_score = 1/(1+10**((opp_rating - org_rating)/400))
    new_rating = org_rating + k*(score - E_score)
    return new_rating


def update_elos(results,white_black_list,engine_elo,net_elo):
    # results is a list of 1,0,-1 for each game
    # white_black_list is a list of booleans for each game (True if engine is white, False if black)
    assert len(results) == len(white_black_list)
    for i in range(len(results)):
        res = results[i]
        if (res == 1 and white_black_list[i]) or (res == -1 and not white_black_list[i]):
            score = 1
        elif (res == -1 and white_black_list[i]) or (res == 1 and not white_black_list[i]):
            score = 0
        else:
            score = 0.5
        net_elo = update_elo(net_elo, engine_elo, score)
    return net_elo

def find_network_elo(network, num_games=100,num_runs=1600,save_pgn_path=None,engine_ratings=[300,500,700,900,1300,1700,2100,2500,2900,3300]):
    """
    Calculates the elo of a network
    Plays num_games/10 against 10 different stockfish versions 
    """
    num_games = int(num_games/len(engine_ratings))
    net_rating = 800
    pgns = []
    for eng_rating in engine_ratings: 
        print('Playing against stockfish with rating: ', eng_rating)
        #play num_games against stockfish with rating
        white_list = [True if i%2 == 0 else False for i in range(num_games)]
        if mp.cpu_count() < num_games:
            raise Exception('Not enough CPU cores to run in parallel')
        pool = mp.Pool(processes=num_games)
        inputs = [(network, eng_rating, white_list[i], num_runs, 0.7,True) for i in range(pool._processes)]
        results = pool.starmap(play, inputs)
        pool.close()
        pool.join()
        scores = [res[0] for res in results]
        pgn = [res[1] for res in results]
        net_rating = update_elos(scores,white_list,eng_rating,net_rating)
        pgns.append(pgn)
    if save_pgn_path is not None:
        file = open(save_pgn_path, 'w')
        for idx,pgn_list in enumerate(pgns):
            file.write('Games against stockfish with rating: ' + str(engine_ratings[idx]) + '\n')
            for game_id,game in enumerate(pgn_list):
                file.write('Game ' + str(game_id) + '\n')
                exporter = PGN.StringExporter(headers=True, variations=True, comments=True)
                file.write(game.accept(exporter) + '\n')
        file.close()
    return net_rating

def evaluate_network(old_network, new_network, num_games=400):
    """
    Evaluates a new network against the previous network
    Returns True if the new network wins 55% of the games
    """
    pass

# if __name__ == '__main__':
    # Testing
    # net = torch.load('final_net')
    # find_network_elo(net,num_runs=10,num_games=4,save_pgn_path='pgn.txt',engine_ratings=[300,500])