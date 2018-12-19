import tensorflow as tf

from agents.ai_agent import AIAgent
from agents.q_agent import QAgent
from agents.human_agent import HumanAgent

import environment as tictac


# Parameters
# ==================================================

# Model directory
tf.flags.DEFINE_string("model_dir", "", "Provide a trained model path if you want to play against a deep agent (default: None)")

FLAGS = tf.flags.FLAGS

def main(argv=None):

    # Initializing the environment
    game = tictac.TicTacToeGame(2,tictac.LoggerLevels.DEBUG)

    # Initialize agents
    agents = []
    agents.append(HumanAgent())

    if FLAGS.model_dir:
        agent = QAgent()
        agent.load_model(FLAGS.model_dir)
        agent.make_greedy()
        agents.append(agent)
    else:
        agent = AIAgent()
        agents.append(agent)

    game.reset()
    keep_playing = True

    while keep_playing:

        players_order = game.get_players_order()
        for player_id in players_order:

            agent = agents[player_id]
            # agent observes state before acting
            agent.observe(game, player_id)
            available_actions = game.get_player_actions(game.board, player_id)
            action = agent.select_action(available_actions)

            game.play_step(action, player_id)

            win, draw = game.evaluate_step(player_id)

            if win:
                print ("player ", player_id, " wins!!")
                keep_playing = False
                break
            elif draw:
                print ("draw!!!")
                keep_playing = False
                break

    game.print_board(game.board)



if __name__ == '__main__':
    tf.app.run()

