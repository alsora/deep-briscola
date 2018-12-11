import numpy as np


class RandomAgent:

    def __init__(self):
        self.observed_state = {}

    def observe(self, *_):
        pass

    def select_action(self, actions):

        return np.random.choice(actions)


    def update(self, reward):
        pass
