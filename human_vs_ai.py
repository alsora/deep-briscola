import itertools, time, random
import numpy as np

import sys
from matplotlib import pyplot as plt

from agents.random_agent import RandomAgent
from agents.naive_agent import NaiveAgent
from agents.human_agent import HumanAgent
import environment as brisc
from rendering import Visualizer



if __name__ == "__main__":

    # Initializing the environment
    game = brisc.BriscolaGame(  summary_turn= True)

    visualizer = Visualizer()
    #visualizer.create_deck(game.deck.deck)

    # Initialize agents
    agents = []
    agents.append(HumanAgent())
    agents.append(NaiveAgent())

    # First reset of the environment
    briscola = game.reset()
    first_turn = True

    while 1:
        print ("-------------- NEW TURN ---------------")
        if not first_turn:
            if not game.draw_step():
                game.end_game()
                break
        first_turn = False

        players_order = game.get_players_order()
        for player_id in players_order:

            player = game.players[player_id]
            agent = agents[player_id]

            agent.observe_game_state(game)
            agent.observe_player_state(player)

            available_actions = game.get_player_actions(player_id)

            action = agent.select_action(available_actions)

            game.play_step(action, player_id)


        winner, points = game.evaluate_step()
