import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import environment as brisc
from train import evaluate

from agents.random_agent import RandomAgent
from agents.ai_agent import AIAgent
from agents.q_agent import QAgent


def stats_plotter(agents, points, total_wins):
    num_evaluations = len(points[0])
    colors = ['green', 'lightblue']

    for i in range(len(agents)):
        plt.figure(figsize = (10,6))
        res = plt.hist(points[i], bins=15, edgecolor = 'black', color = colors[i],
            label = agents[i].name + " " + str(i) + " points")
        plt.title(agents[i].name + " " + str(i) + " won {:.2%}".format(total_wins[i]/num_evaluations))
        plt.vlines(np.mean(points[i]),
            0,
            max(res[0])/10,
            label = 'Points mean',
            color = 'black',
            linewidth = 3)
        plt.vlines([np.mean(points[i]) - np.std(points[i]),
            np.mean(points[i]) + np.std(points[i])],
            ymin=0,
            ymax=max(res[0])/10,
            label = 'Points mean +- std',
            color = 'red',
            linewidth = 3)
        plt.xlim(0,120); plt.legend(); plt.show()


def main(argv=None):

    game = brisc.BriscolaGame(2,verbosity=brisc.LoggerLevels.TRAIN)

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
    tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
    tf.flags.DEFINE_integer("num_evaluations", 20, "Number of training epochs")

    FLAGS = tf.flags.FLAGS

    tf.app.run()








