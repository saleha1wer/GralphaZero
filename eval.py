from play import play
import torch 
import multiprocessing as mp
import chess.pgn as PGN
from old_network import Network as oldNetwork
from network import Network
import chess 
from play import get_move
from collections import Counter
import copy
import numpy as np
from play import _print_pgn
from gui import GUI
def update_elo(org_rating, opp_rating,score, k=32):
    """
    Updates the elo of the winner and loser
    """
    E_score = 1/(1+10**((opp_rating - org_rating)/400))
    new_rating = org_rating + k*(score - E_score)
    return new_rating


def update_elos(results,white_black_list,engine_elo,net_elo,show=False):
    # results is a list of 1,0,-1 for each game
    # white_black_list is a list of booleans for each game (True if engine is white, False if black)
    assert len(results) == len(white_black_list)
    wins,losses,draws = 0,0,0

    for res,white in zip(results,white_black_list):
        if (res == 1 and white) or (res == -1 and not white):
            score = 1
            wins += 1
        elif (res == -1 and white) or (res == 1 and not white):
            score = 0
            losses += 1
        else:
            score = 0.5
            draws += 1

        net_elo = update_elo(net_elo, engine_elo, score)
    if show:
        print('Results against stockfish with rating: ', engine_elo, ' are:')
        print('Wins: ', wins, ' Losses: ', losses, ' Draws: ', draws)
        print('New elo: ', net_elo)

    return net_elo

def find_network_elo(network,gui=True,exploration=False,multi=True,from_mcts=True, num_games=100,num_runs=1600,c=0.5,save_pgn_path=None,engine_ratings=[300,500,700,900,1300,1700,2100,2500,2900,3300],net_elo=800,show_update=True,show_pgn=True,depth=None,level=None):
    """
    Calculates the elo of a network
    Plays num_games/10 against 10 different stockfish versions 
    """
    # score,pgn = play(network,engine_ratings[0],True,num_runs=num_runs,c=c,return_pgn=True,from_mcts=from_mcts)
    # print(pgn)
    num_games = int(num_games/len(engine_ratings))
    net_rating = net_elo
    pgns = []
    for eng_rating in engine_ratings: 
        # score,pgn = play(network, eng_rating, white=True, num_runs=num_runs, c=c,return_pgn=True,from_mcts=from_mcts,exploration=False)
        print('Playing against stockfish with rating: ', eng_rating)
        #play num_games against stockfish with rating
        white_list = [True if i%2 == 0 else False for i in range(num_games)]
        if multi:
            if mp.cpu_count() < num_games:
                raise Exception('Not enough CPU cores to run in parallel')
            pool = mp.Pool(processes=num_games)
            network.eval()
            inputs = [(network, eng_rating, white_list[i], num_runs, c,True,from_mcts,exploration,show_pgn,depth,level) for i in range(pool._processes)]
            results = pool.starmap(play, inputs)
            pool.close()
            pool.join()
            scores = [res[0] for res in results]
            pgn = [res[1] for res in results]
            net_rating = update_elos(scores,white_list,eng_rating,net_rating,show=show_update)
            pgns.append(pgn)
        else:
            scores,pgn,gui_infos = [],[],[]
            for i in range(num_games):
                if i%10 == 0:
                    print(f'Playing game {i+1}/{num_games} against Stockfish with rating: {eng_rating}')
                network.eval()
                if gui:
                    score, game_pgn,gui_info = play(network, eng_rating, white_list[i], num_runs, c, True, from_mcts, exploration,show_pgn=show_pgn,depth=depth,level=level,gui=gui)
                    gui_infos.append(gui_info)
                else:
                    score, game_pgn = play(network, eng_rating, white_list[i], num_runs, c, True, from_mcts, exploration,show_pgn=show_pgn,depth=depth,level=level,gui=gui)
                scores.append(score)
                pgn.append(game_pgn)
            net_rating = update_elos(scores,white_list,eng_rating,net_rating,show=show_update)
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
    if gui:
        gui = GUI()
        for game in gui_infos:
            gui.add_game(game)
        gui.run()
    return net_rating

def evaluate_network(old_network, new_network, num_games=400,from_mcts=True,num_runs=800,names=['Old Network','New Network'],c=0.7,exploration=False,show=True):
    """
    Evaluates a new network against the previous network
    Returns True if the new network wins 55% of the games
    """
    white_list = [True if i%2 == 0 else False for i in range(num_games)]
    old_network.eval()
    new_network.eval()
    old_white_results = []
    new_white_results = []
    pgns = []
    for i in range(1,num_games+1):
        board = chess.Board()
        print('Game: ', i)
        while board.is_game_over() is False:
            if show:
                print(board)
            white_net = old_network if white_list[i-1] else new_network
            black_net = new_network if white_list[i-1] else old_network
            net = white_net if board.turn else black_net
            # from_mcts = True if board.turn == chess.BLACK else False
            move,_ = get_move(net,board,c,num_runs,from_mcts,exploration=exploration)
            board.push(move)
            if show:
                _print_pgn(board,name1=names[0],name2=names[1])
        pgn = PGN.Game().from_board(board)
        pgn.headers['White'] = names[0] if white_list[i-1] else names[1]
        pgn.headers['Black'] = names[1] if white_list[i-1] else names[0]
        exporter = PGN.StringExporter(headers=True, variations=True, comments=True)
        print(pgn.accept(exporter))
        pgns.append(copy.deepcopy(pgn.accept(exporter)))
        if white_list[i-1]:
            old_white_results.append(board.result())
        else:
            new_white_results.append(board.result())
    for idx,i in enumerate(pgns):
        print('PGN game ', idx+1, ' :')
        print(i)
    old_wins = old_white_results.count('1-0') + new_white_results.count('0-1')
    new_wins = old_white_results.count('0-1') + new_white_results.count('1-0')
    draws = old_white_results.count('1/2-1/2') + new_white_results.count('1/2-1/2')
    return old_wins, new_wins, draws


def play_two_nets(net1_config, net2_config, net1_params_path,net2_params_path, net1_name, net2_name,num_games=10,from_mcts=True,num_runs=777,c=0.3,explr = False,show=True):
    net1 = Network(net1_config,name='new')
    net2 = Network(net2_config,name='new')
    for net, params in zip([net1,net2],[net1_params_path,net2_params_path]):
        net.load_state_dict(torch.load(params))
        net.eval()
    net1wins, net2wins, draws = evaluate_network(net1, net2, num_games=num_games,from_mcts=from_mcts,num_runs=num_runs,names=[net1_name,net2_name],c=c,exploration=explr,show=show)
    
    print(net1_name, ' Wins: ')
    print(net1wins)
    print(net2_name, ' Wins: ')
    print(net2wins)
    print('Draws')
    print(draws)


def play_against_net(network_config,network_params,white=True,from_mcts=True,nruns=800):
    network = Network(network_config,name='new')
    network.load_state_dict(torch.load(network_params))
    network.eval()
    game = chess.Board()

    while game.outcome() is None:
        print(game.fen())
        if white == game.turn:
            move,_ = get_move(network,game,0.7,nruns,from_mcts=from_mcts,exploration=False)
            print(move)
        else:
            move = input('Enter move: ')
            move = chess.Move(chess.parse_square(move[0:2]),chess.parse_square(move[2:4]))
            while move not in game.legal_moves:
                print('Illegal move')
                move = input('Enter move: ')
                move = chess.Move(chess.parse_square(move[0:2]),chess.parse_square(move[2:4]))
            # move = np.random.choice(list(game.legal_moves))
        game.push(move)
    pgn = PGN.Game().from_board(game)
    exporter = PGN.StringExporter(headers=True, variations=True, comments=True)
    print(pgn.accept(exporter))
def getstrbet(s, start, end):
    return s[s.index(start) + len(start):s.rindex(end)]

if __name__ == '__main__':  
    # Testing

    print('Loading network params')

    config1 = {'att_emb_size':515,'GAheads':5,'loss_func':None,'lr': 0.0001 , 'hidden': 4672, 'n_layers': 2,'heads': 16,'gnn_type':'GAT','board_representation':'graph','useresnet':False}
    params1 = 'new_networks/new_graph_6k400k_params'
    name1 = 'Old Graph ({})'.format(getstrbet(params1,'new_networks/new_graph_','_params'))

    config2 = {'att_emb_size':128,'GAheads':32,'loss_func':None,'lr': 0.0001 , 'hidden': 515, 'n_layers': 6,'heads': 32,'gnn_type':'GAT','board_representation':'graph','useresnet':False}
    params2 ='new_networks/graph_6k400k_params'
    name2 = 'New Graph ({})'.format(getstrbet(params2,'graph_','_params'))
    
    # from_mcts, explr,show = True, False, True
    from_mcts, explr,show = False, True, False
    play_two_nets(config1,config2,params1,params2,name1,name2,from_mcts=from_mcts, num_runs=150,num_games=1000,c=1.5,explr=explr,show=show)

    net = Network(config1)
    net.load_state_dict(torch.load(params1))
    net.eval()
    net.name= 'new'
    # play_against_net(config1,params1,white=True,from_mcts=False,nruns=200)
    # find_network_elo(net,gui=False,depth=1,level=None,multi=False,from_mcts=False,num_runs=700,num_games=20,c=1.6,save_pgn_path='pgn.txt',engine_ratings=[10],net_elo=10,show_pgn=True,exploration=False,show_update=True)

