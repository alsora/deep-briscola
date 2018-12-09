import itertools, time, random
import numpy as np

import sys
from matplotlib import pyplot as plt

from agents.random_agent import RandomAgent
from agents.naive_agent import NaiveAgent
from agents.deep_agent import DeepAgent
from agents.human_agent import HumanAgent
import environment as brisc
from rendering import Visualizer




def main(argv):

    # Initializing the environment
    game = brisc.BriscolaGame(brisc.LoggerLevels.DEBUG)
    deck = game.deck

    visualizer = Visualizer()
    #visualizer.create_deck(game.deck.deck)

    # Initialize agents
    agents = []
    agents.append(HumanAgent())
    agents.append(DeepAgent())

    agents[1].load_model()

    # First reset of the environment
    briscola = game.reset()
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





if __name__ == "__main__":
    main(sys.argv)

