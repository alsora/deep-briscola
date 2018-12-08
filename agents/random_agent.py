import numpy as np


class RandomAgent:

    def __init__(self):
        self.observed_state = {}

    def observe_game_state(self, game = None, deck = None):
        pass

    def observe_player_state(self, player = None, deck = None):
        pass

    def select_action(self, actions):

        return np.random.choice(actions)


    def update(self, reward, new_state):
        pass
