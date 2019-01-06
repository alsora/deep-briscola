import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

from agents.random_agent import RandomAgent
from agents.ai_agent import AIAgent
from agents.q_agent import QAgent
from graphic_visualizations import stats_plotter
import environment as brisc
from utils import BriscolaLogger


def evaluate(game, agents, num_evaluations):

    total_wins = [0] * len(agents)
    points_history = [ [] for i in range(len(agents))]

    for _ in range(num_evaluations):

        game_winner_id, winner_points = brisc.play_episode(game, agents, train=False)

        for player in game.players:
            points_history[player.id].append(player.points)
            if player.id == game_winner_id:
                total_wins[player.id] += 1

    print("\nTotal wins: ",total_wins)
    for i in range(len(agents)):
        print(agents[i].name + " " + str(i) + " won {:.2%}".format(total_wins[i]/num_evaluations), " with average points {:.2f}".format(mean(points_history[i])))

    return total_wins, points_history



def main(argv=None):
    '''Evaluate agent performances against RandomAgent and AIAgent'''

    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TEST)
    game = brisc.BriscolaGame(2, logger)

    # Initialize agents
    agents = []

    # first agent is RandomAgent or QAgent if a model is provided
    if FLAGS.model_dir:
        agent = QAgent()
        agent.load_model(FLAGS.model_dir)
        agent.make_greedy()
        agents.append(agent)
    else:
        agent = RandomAgent()
        agents.append(agent)

    # test first agent against RandomAgent
    agents.append(RandomAgent())

    total_wins, points_history = evaluate(game, agents, FLAGS.num_evaluations)
    stats_plotter(agents, points_history, total_wins)

    # test first agent against AIAgent
    agents[1] = AIAgent()

    total_wins, points_history = evaluate(game, agents, FLAGS.num_evaluations)
    stats_plotter(agents, points_history, total_wins)



if __name__ == '__main__':

    # Parameters
    # ==================================================

    # Model directory
    tf.flags.DEFINE_string("model_dir", "", "Provide a trained model path if you want to play against a deep agent (default: None)")

    # Training parameters
    tf.flags.DEFINE_integer("num_evaluations", 20, "Number of training epochs")

    FLAGS = tf.flags.FLAGS

    tf.app.run()








