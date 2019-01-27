import tensorflow as tf
import argparse
import numpy as np
from statistics import mean

from agents.random_agent import RandomAgent
from agents.ai_agent import AIAgent
from agents.q_agent import QAgent
from graphic_visualizations import stats_plotter
import environment as brisc
from utils import BriscolaLogger
from utils import NetworkTypes


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

    # agent to be evaluated is RandomAgent or QAgent if a model is provided
    if FLAGS.model_dir:
        eval_agent = QAgent()
        eval_agent.load_model(FLAGS.model_dir)
        eval_agent.make_greedy()
    else:
        eval_agent = RandomAgent()

    # test agent against RandomAgent
    agents = [eval_agent, RandomAgent()]

    total_wins, points_history = evaluate(game, agents, FLAGS.num_evaluations)
    stats_plotter(agents, points_history, total_wins)

    # test agent against AIAgent
    agents = [eval_agent, AIAgent()]

    total_wins, points_history = evaluate(game, agents, FLAGS.num_evaluations)
    stats_plotter(agents, points_history, total_wins)



if __name__ == '__main__':

    # Parameters
    # ==================================================

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default=None, help="Provide a trained model path if you want to play against a deep agent", type=str)
    parser.add_argument("--network", default=NetworkTypes.DRQN, choices=[NetworkTypes.DQN, NetworkTypes.DRQN], help="Neural Network used for approximating value function")
    parser.add_argument("--num_evaluations", default=20, help="Number of evaluation games against each type of opponent for each test", type=int)

    FLAGS = parser.parse_args()

    tf.app.run()








