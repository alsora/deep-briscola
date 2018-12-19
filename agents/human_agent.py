import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import environment as tictac

class HumanAgent:

    def __init__(self):
        pass

    def observe(self, game, player_id):
        self.board = game.board
        self.id = player_id


    def select_action(self, actions):
        ''' parse user input from keyboard
            if it's not a valid action index, do something random
        '''

        print("Your turn!")
        print ("Your value is ", self.id)

        tictac.TicTacToeGame.print_board(self.board)

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
