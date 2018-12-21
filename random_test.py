import tensorflow as tf
import environment as brisc
import matplotlib.pyplot as plt
import numpy as np

from agents.random_agent import RandomAgent
from agents.ai_agent import AIAgent



def stats_gathering(agents, N):
    # Initializing the environment
    game = brisc.BriscolaGame(2,verbosity=brisc.LoggerLevels.TRAIN)
    deck = game.deck

    # Statistics
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

    return point_0, winner_0, point_1, winner_1


def stats_plotter(agents, point_0, winner_0, point_1, winner_1):
    N = len(point_0)
    
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


def main(argv=None):
    
    # Random agent vs AI hard coded agent
    RandomVsAi = []
    RandomVsAi.append(RandomAgent())
    RandomVsAi.append(AIAgent())
    # Stats 
    N = 1000
    point_0, winner_0, point_1, winner_1 = stats_gathering(RandomVsAi, N)
    stats_plotter(RandomVsAi, point_0, winner_0, point_1, winner_1)
    

    # Random agent vs random agent
    RandomVsRandom = []
    RandomVsRandom.append(RandomAgent())
    RandomVsRandom.append(RandomAgent())
    # Stats 
    N = 1000
    point_0, winner_0, point_1, winner_1 = stats_gathering(RandomVsRandom, N)
    stats_plotter(RandomVsRandom, point_0, winner_0, point_1, winner_1)    




if __name__ == '__main__':
    main()









