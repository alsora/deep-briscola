import itertools, time, random
import numpy as np

import sys
from matplotlib import pyplot as plt

from agents.random_agent import RandomAgent
from agents.deep_agent import DeepAgent
import environment as brisc


def test(game, agents):

    deck = game.deck
    n_games = 100
    total_wins = [0, 0]
    total_points = [0, 0]

    for _ in range(n_games):

        game.reset()

        while 1:
            deck = game.deck
            players_order = game.get_players_order()
            for player_id in players_order:

                player = game.players[player_id]
                agent = agents[player_id]

                agent.observe(game, player, deck)
                available_actions = game.get_player_actions(player_id)
                action = agent.select_action(available_actions)

                game.play_step(action, player_id)

            winner_player_id, points = game.evaluate_step()

            if not game.draw_step():
                game_winner_id, winner_points = game.end_game()

                total_wins[game_winner_id] += 1
                total_points[game_winner_id] += winner_points
                total_points[1 - game_winner_id] += (120 - winner_points)
                break


    print("DeepAgent wins ", total_wins[0], "% with average points ", float(total_points[0])/float(n_games))





if __name__ == "__main__":

    # Initializing the environment
    game = brisc.BriscolaGame(  summary_turn= True)
    deck = game.deck

    # Initialize agents
    agents = []
    agents.append(DeepAgent())
    agents.append(RandomAgent())


    train_epochs = 50000
    test_every = 2500

    # First reset of the environment
    game.reset()

    for epoch in range(0, train_epochs):
        print ("Epoch: ", epoch)
        game.reset()

        while 1:

            # step
            players_order = game.get_players_order()
            for player_id in players_order:

                player = game.players[player_id]
                agent = agents[player_id]

                agent.observe(game, player, deck)
                available_actions = game.get_player_actions(player_id)
                action = agent.select_action(available_actions)

                game.play_step(action, player_id)


            winner_player_id, points = game.evaluate_step()

            # TODO: should update also after last hand
            # update environment
            if not game.draw_step():
                game_winner_id, winner_points = game.end_game()
                break

            # update agents
            for player_id in players_order:
                player = game.players[player_id]
                agent = agents[player_id]

                agent.observe(game, player, deck)
                available_actions = game.get_player_actions(player_id)

                # compute reward function for this player
                if player_id is winner_player_id:
                    reward = 2
                else:
                    reward = -2
                if points >= 10:
                    reward *= 2

                agent.update(reward, available_actions)


        if epoch % test_every == 0:
            test(game, agents)


