import tensorflow as tf

from agents.ai_agent import AIAgent
from agents.q_agent import QAgent
from agents.human_agent import HumanAgent

import environment as brisc


# Parameters
# ==================================================

# Model directory
tf.flags.DEFINE_string("model_dir", "", "Provide a trained model path if you want to play against a deep agent (default: None)")

FLAGS = tf.flags.FLAGS

def main(argv=None):

    # Initializing the environment
    game = brisc.BriscolaGame(2,brisc.LoggerLevels.DEBUG)
    deck = game.deck

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

    # First reset of the environment
    briscola = game.reset()
    keep_playing = True

    while keep_playing:

        players_order = game.get_players_order()
        for player_id in players_order:

            player = game.players[player_id]
            agent = agents[player_id]

            agent.observe(game, player, deck)
            available_actions = game.get_player_actions(player_id)
            action = agent.select_action(available_actions)

            game.play_step(action, player_id)

        winner_player_id, points = game.evaluate_step()

        keep_playing = game.draw_step()


    game_winner_id, winner_points = game.end_game()





if __name__ == '__main__':
    tf.app.run()

