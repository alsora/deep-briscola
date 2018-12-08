import itertools, time, random
import numpy as np

import sys
from matplotlib import pyplot as plt

from agents.random_agent import RandomAgent
from agents.deep_agent import DeepAgent
import environment as brisc


def test(game, agents):

    n_games = 1
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

                agent.observe_game_state(game, deck)
                agent.observe_player_state(player, deck)
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

    # Initialize agents
    agents = []
    agents.append(DeepAgent())
    agents.append(RandomAgent())

    # First reset of the environment
    game.reset()

    while 1:

        deck = game.deck
        players_order = game.get_players_order()
        for player_id in players_order:

            player = game.players[player_id]
            agent = agents[player_id]

            agent.observe_game_state(game, deck)
            agent.observe_player_state(player, deck)
            available_actions = game.get_player_actions(player_id)
            action = agent.select_action(available_actions)

            game.play_step(action, player_id)


        winner_player_id, points = game.evaluate_step()

        # TODO: should update also after last hand
        if not game.draw_step():
            game_winner_id, winner_points = game.end_game()
            break

        for player_id in players_order:
            player = game.players[player_id]
            agent = agents[player_id]

            agent.observe_game_state(game, deck)
            agent.observe_player_state(player, deck)
            available_actions = game.get_player_actions(player_id)

            # compute reward function for this player
            if player_id is winner_player_id:
                reward = 2
            else:
                reward = -2
            if points >= 10:
                reward *= 2

            agent.update(reward, available_actions)


    test(game, agents)



