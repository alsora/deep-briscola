import numpy as np


class HumanAgent:

    def __init__(self):
        self.observed_state = {}

    def observe(self, game, player, deck):
        self.observed_state['hand'] = player.hand
        self.observed_state['briscola'] = game.briscola
        self.observed_state['played_cards'] = game.played_cards

    def select_action(self, actions):

        print("Your turn!")
        print ("The briscola is ", self.observed_state['briscola'].name)
        print ("Your hand is: ", [card.name for card in self.observed_state['hand']])

        try:
            action=int(input('Input:'))
        except ValueError:
            print("Not a number")
            action = np.random.choice(actions)

        if action not in actions:
            print ("Error, selected out of bounds action!!")
            action = np.random.choice(actions)

        return action


    def update(self, reward):
        pass
