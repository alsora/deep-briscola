import tensorflow as tf

from agents.random_agent import RandomAgent
import environment as tictac



def main(argv=None):
    # Initializing the environment
    game = tictac.TicTacToeGame(verbosity=tictac.LoggerLevels.DEBUG)

    # Initialize agents
    agents = []
    agents.append(RandomAgent())
    agents.append(RandomAgent())

    # First reset of the environment
    game.reset()
    keep_playing = True
    winner = -1
    draw = False
    while keep_playing:

        # action step
        players_order = game.get_players_order()
        rewards = [0, 0]
        for player_id in players_order:

            agent = agents[player_id]
            # agent observes state before acting
            agent.observe(game, player_id)
            available_actions = game.get_player_actions(game.board, player_id)
            action = agent.select_action(available_actions)

            game.play_step(action, player_id)

            if game.check_winner(game.board, player_id):
                keep_playing = False
                winner = player_id
                break
            elif game.check_board_full(game.board):
                keep_playing = False
                draw = True
                break

    if draw:
        print ("Game ended with a draw!!")
    else:
        print ("Player ", winner, " wins the game!!")

    game.print_board(game.board)


if __name__ == '__main__':
    tf.app.run()
