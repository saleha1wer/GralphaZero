"""
1. Self-Play for n games (MCTS)
    - Store all UCB move-selections for past 20*n games in buffer
2. Train network on a random batch of the past 20*n games for 1 epoch
3. Repeat 
- Every 100 loops, calc elo of network
- Every 1,000 loops, Evaluate against previous network (for 400 games) and save if wins 55% of games
- In alphazero, n = 25,000 and batch_size = 2048
"""