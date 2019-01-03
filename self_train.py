import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

## our stuff import
import graphic_visualizations as gv
import environment as brisc
from evaluate import evaluate

from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.ai_agent import AIAgent


class CopyAgent(QAgent):
    '''Copied agent. Identical to a QAgent, but does not update itself'''
    def update(self, *args):
        pass



def self_train(game, agent, num_epochs, evaluate_every, num_evaluations, random_every, model_dir = "", evaluation_dir = "evaluation_dir"):

    # Initializing the list of agents with the agent and a copy of him
    if not os.path.isdir('cur_model_copy'):
        os.makedirs('cur_model_copy')
    agent.save_model('cur_model_copy')
    first_old = CopyAgent()
    first_old.load_model('cur_model_copy')
    old_agents = [agent, first_old]

    # Training starts
    best_winning_ratio = -1
    for epoch in range(1, num_epochs + 1):

        # picking an agent from the past self
        old_num = np.random.randint(1,len(old_agents))
        agents= [agent, old_agents[len(old_agents)-old_num]]

        gv.printProgressBar(epoch, num_epochs,
                            prefix = f'Epoch: {epoch}',
                            length= 50)

        # During the playing of the episode the agent learn
        brisc.play_episode(game, agents)

        # Evaluation step
        if epoch % evaluate_every == 0:
            for ag in agents:
                ag.make_greedy()
            victory_rates, average_points = evaluate(game, agents, num_evaluations)
            victory_rates_hist.append(victory_rates)
            average_points_hist.append(average_points)

            # EVALUATION VISUALISATION
            if not os.path.isdir(evaluation_dir):
                os.mkdir(evaluation_dir)
            output_path = evaluation_dir + "/fig_" + str(epoch)
            std_cur = gv.eval_visua_for_self_play(average_points_hist,
                             FLAGS,
                             victory_rates_hist,
                             output_path=output_path)
            # Storing std
            std_hist.append(std_cur)

            for ag in agents:
                ag.restore_epsilon()

            # After the evaluation we add the agent to the old agents
            agent.save_model('cur_model_copy')
            new_old_agent = CopyAgent()
            new_old_agent.load_model('cur_model_copy')
            old_agents.append(new_old_agent)

            # Eliminating the oldest agent if maximum number of agents
            if len(old_agents) > FLAGS.max_old_agents:
                old_agents.pop(0)

        # Evaluation against random agent
        if epoch % random_every == 0:
            if not os.path.isdir(evaluation_dir):
                os.mkdir(evaluation_dir)
            agents = [agent, RandomAgent()]
            for ag in agents:
                ag.make_greedy()

            winners, points = evaluate(game, agents, FLAGS.num_evaluations)
            output_prefix = evaluation_dir + '/againstRandom_' + str(epoch)
            gv.stats_plotter(agents, points, winners, output_prefix=output_prefix)

            for ag in agents:
                ag.restore_epsilon()

            # Saving the model if the agent performs better against random agent
            if winners[0] > best_winning_ratio:
                best_winning_ratio = winners[0]
                agent.save_model(model_dir)

    return best_winning_ratio



def main(argv=None):

    global victory_rates_hist
    victory_rates_hist  = []
    global average_points_hist
    average_points_hist = []
    global std_hist
    std_hist = []

    global victory_rates_hist_against_Random
    victory_rates_hist_against_Random  = []
    global average_points_hist_against_Random
    average_points_hist_against_Random = []
    global std_hist_against_Random
    std_hist_against_Random = []

    # Initializing the environment
    game = brisc.BriscolaGame(2, verbosity=brisc.LoggerLevels.TRAIN)

    # Initialize agent
    agent = QAgent(
        FLAGS.epsilon, FLAGS.epsilon_increment, FLAGS.epsilon_max, FLAGS.discount,
        FLAGS.learning_rate)

    # Training
    best_winning_ratio = self_train(game, agent,
                                    FLAGS.num_epochs,
                                    FLAGS.evaluate_every,
                                    FLAGS.num_evaluations,
                                    FLAGS.against_random_every,
                                    FLAGS.model_dir,
                                    FLAGS.evaluation_dir)
    print(f'Best winning ratio : {best_winning_ratio}')
    # SUMMARY GRAPH
    gv.summ_vis_self_play(victory_rates_hist, std_hist, FLAGS)



if __name__ == '__main__':

    # Parameters
    # ==================================================

    # Directories
    tf.flags.DEFINE_string("model_dir", "saved_model", "Where to save the trained model, checkpoints and stats (default: pwd/saved_model)")
    tf.flags.DEFINE_string("evaluation_dir", "evaluation_dir3", "Where to save the trained model, checkpoints and stats (default: pwd/saved_model)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
    tf.flags.DEFINE_integer("num_epochs", 100000, "Number of training epochs")
    tf.flags.DEFINE_integer("max_old_agents", 100, "Maximum number of old copies of self stored")

    # Deep Agent parameters
    tf.flags.DEFINE_float("epsilon", 0, "How likely is the agent to choose the best reward action over a random one (default: 0)")
    tf.flags.DEFINE_float("epsilon_increment", 5e-5, "How much epsilon is increased after each action taken up to epsilon_max (default: 5e-6)")
    tf.flags.DEFINE_float("epsilon_max", 0.85, "The maximum value for the incremented epsilon (default: 0.85)")
    tf.flags.DEFINE_float("discount", 0.85, "How much a reward is discounted after each step (default: 0.85)")

    # Network parameters
    tf.flags.DEFINE_float("learning_rate", 1e-4, "The learning rate for the network updates (default: 1e-4)")


    # Evaluation parameters
    tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model after this many steps (default: 1000)")
    tf.flags.DEFINE_integer("against_random_every", 2000, "Evaluate model after this many steps (default: 1000)")
    tf.flags.DEFINE_integer("num_evaluations", 200, "Evaluate on these many episodes for each test (default: 500)")

    FLAGS = tf.flags.FLAGS

    tf.app.run()
























































