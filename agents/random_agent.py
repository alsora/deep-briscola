import numpy as np


class RandomAgent:

    def __init__(self):
        self.observed_state = {}

    def observe(self, game, player, deck):
        pass

    def select_action(self, actions):

        return np.random.choice(actions)


    def update(self, reward, new_state):
        pass
