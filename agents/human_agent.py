import numpy as np


class HumanAgent:

    def __init__(self):
        self.name = 'HumanAgent'

    def observe(self, game, player_id):
        player = game.players[player_id]
        self.hand = player.hand
        self.briscola = game.briscola
        self.played_cards = game.played_cards

    def select_action(self, actions):
        ''' parse user input from keyboard
            if it's not a valid action index, do something random
        '''

        print("Your turn!")
        print ("The briscola is ", self.briscola.name)
        print ("Your hand is: ", [card.name for card in self.hand])

        try:
            action=int(input('Input:'))
        except ValueError:
            print("Not a number")
            action = np.random.choice(actions)

        if action not in actions:
            print ("Error, selected out of bounds action!!")
            action = np.random.choice(actions)

        return action


    def store_experience(self, reward):
        pass

    def make_greedy(self):
        pass

    def restore_epsilon(self):
        pass
