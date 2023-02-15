from play import play
import torch 
def update_elo(org_rating, opp_rating,score, k=32):
    """
    Updates the elo of the winner and loser
    """
    E_score = 1/(1+10**((opp_rating - org_rating)/400))
    new_rating = org_rating + k*(score - E_score)
    return new_rating

def find_network_elo(network, num_games=100):
    """
    Calculates the elo of a network
    Plays num_games/10 against 10 different stockfish versions 
    """
    engine_ratings = [300,500,700,900,1300,1700,2100,2500,2900,3300]
    num_games = int(num_games/len(engine_ratings))
    net_rating = 800
    for eng_rating in engine_ratings: 
        print('Playing against stockfish with rating: ', eng_rating)
        #play num_games against stockfish with rating
        white = True
        for i in range(num_games):
            res = play(network,eng_rating,white=white,c=0.5,num_runs=700)
            if (res == 1 and white) or (res == -1 and not white):
                score = 1
            elif (res == -1 and white) or (res == 1 and not white):
                score = 0
            else:
                score = 0.5
            net_rating = update_elo(net_rating, eng_rating, score)
            print('Score: ', score)
            print('Network rating: ', net_rating)
            white = not white
    return net_rating

def evaluate_network(old_network, new_network, num_games=400):
    """
    Evaluates a new network against the previous network
    Returns True if the new network wins 55% of the games
    """
    pass

net = torch.load('final_net')
find_network_elo(net)