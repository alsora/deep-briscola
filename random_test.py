import tensorflow as tf
import environment as brisc
import matplotlib.pyplot as plt
import numpy as np

from agents.random_agent import RandomAgent
from agents.ai_agent import AIAgent

def main(argv=None):
    # Initializing the environment
    game = brisc.BriscolaGame(2,verbosity=brisc.LoggerLevels.TRAIN)
    deck = game.deck

    # Initialize agents
    agents = []
    agents.append(RandomAgent())
    agents.append(AIAgent())

    # Statistics
    N = 10000
    winner_0 = 0
    winner_1 = 0
    point_0 = []
    point_1 = []
    
    for _ in range(N):
        
        game.reset()
        keep_playing = True
    
        while keep_playing:
    
            players_order = game.get_players_order()
            for player_id in players_order:
    
                player = game.players[player_id]
                agent = agents[player_id]
    
                agent.observe(game, player, deck)
                available_actions = game.get_player_actions(player_id)
                action = agent.select_action(available_actions)
    
                game.play_step(action, player_id)
    
            winner_player_id, points = game.evaluate_step()
    
            keep_playing = game.draw_step()
    
    
        game_winner_id, winner_points = game.end_game()
        
        if game_winner_id:
            point_1.append(winner_points)
            winner_1 += 1
            point_0.append(120-winner_points)
        else:
            point_0.append(winner_points)
            winner_0 += 1
            point_1.append(120-winner_points)
    
    
    ## STATISTICS OBSERVATION
    plt.figure(figsize = (10,6))
    res = plt.hist(point_1, bins=15, edgecolor = 'black', color = 'green',
             label = 'Player 1 points')
    plt.title(f"Player 1 ({agents[1].name}) won {winner_1/N}% of times")
    plt.vlines(np.mean(point_1),
               0, 
               max(res[0])/10, 
               linewidth = 3,
               label = 'Points mean')
    plt.vlines([np.mean(point_1) - np.std(point_1),
                np.mean(point_1) + np.std(point_1)],
                ymin=0,
                ymax=max(res[0])/10,
                label = 'Points mean +- std', 
                color = 'red',
                linewidth = 3)
    plt.xlim(0,120); plt.legend(); plt.show()
    
    plt.figure(figsize = (10,6))
    res = plt.hist(point_0, bins=15, edgecolor = 'black', color = 'lightblue',
             label = 'Player 0 points')
    plt.title(f"Player 0 ({agents[0].name}) won {winner_0/N}% of times")
    plt.vlines(np.mean(point_0),
               0, 
               max(res[0])/10, 
               linewidth = 3,
               label = 'Points mean')
    plt.vlines([np.mean(point_0) - np.std(point_0),
                np.mean(point_0) + np.std(point_0)],
                ymin=0,
                ymax=max(res[0])/10,
                label = 'Points mean +- std', 
                color = 'red',
                linewidth = 3)
    plt.xlim(0,120); plt.legend(); plt.show()
    

if __name__ == '__main__':
    main()









