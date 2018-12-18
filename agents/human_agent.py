import numpy as np


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

        self.board = [int(i) for i in self.board]
        board_matrix = np.reshape(self.board, (3, 3))
        print('\n'.join([''.join(['{:4}'.format(item) for item in row])
            for row in board_matrix]))

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
