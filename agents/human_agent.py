import numpy as np


class HumanAgent:

    def __init__(self):
        self.observed_state = {}

    def observe_game_state(self, game):
        self.observed_state['turn_state'] = game.get_turn_state()
        self.observed_state['briscola'] = game.briscola
        self.observed_state['briscola_seed'] = game.briscola_seed

    def observe_player_state(self, player):
        self.observed_state['hand'] = player.get_player_state()

    def select_action(self, actions):

        print("Your turn!")
        print ("The briscola is ", self.observed_state['briscola'].name)
        print ("Your hand is: ", [card.name for card in self.observed_state['hand']])

        try:
            action=int(input('Input:'))
        except ValueError:
            print("Not a number")
            action = np.random.choice(actions)

        if action > len(actions):
            print ("Error, selected out of bounds action!!")
            action = np.random.choice(actions)

        return action


    def update(self, reward, new_state):
        pass
