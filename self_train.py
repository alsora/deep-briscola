import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

## our stuff import
import graphic_visualizations as gv
import environment as brisc
from graphic_visualizations import stats_plotter

from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.ai_agent import AIAgent


def play_episode(game, agents):

    game.reset()
    keep_playing = True
    while keep_playing:

        # action step
        players_order = game.get_players_order()
        for player_id in players_order:

            player = game.players[player_id]
            agent = agents[player_id]

            # agent observes state before acting
            agent.observe(game, player, game.deck)
            available_actions = game.get_player_actions(player_id)
            action = agent.select_action(available_actions)

            game.play_step(action, player_id)

        rewards = game.get_rewards_from_step()
        # update agents
        for i, player_id in enumerate(players_order):
            player = game.players[player_id]
            agent = agents[player_id]
            # agent observes new state after acting
            agent.observe(game, player, game.deck)

            reward = rewards[i]
            # We want to update only the real agent, not its past copies
            if agent.name != 'copy':
                agent.update(reward)

        # update the environment
        keep_playing = game.draw_step()

    return game.end_game()



def evaluate(game, agents, num_evaluations):

    total_wins = [0] * len(agents)
    points_history = [ [] for i in range(len(agents))]

    for _ in range(num_evaluations):

        game.reset()
        keep_playing = True

        while keep_playing:

            players_order = game.get_players_order()
            for player_id in players_order:

                player = game.players[player_id]
                agent = agents[player_id]

                agent.observe(game, player, game.deck)
                available_actions = game.get_player_actions(player_id)
                action = agent.select_action(available_actions)

                game.play_step(action, player_id)

            winner_player_id, points = game.evaluate_step()

            keep_playing = game.draw_step()

        game_winner_id, winner_points = game.end_game()

        for player in game.players:
            points_history[player.id].append(player.points)
            if player.id == game_winner_id:
                total_wins[player.id] += 1

    return total_wins, points_history



def self_train(game, agent, num_epochs, evaluate_every, num_evaluations, random_every, model_dir = "", evaluation_dir = "evaluation_dir"):

    # Initializing the list of agents with the agent and a copy of him
    if not os.path.isdir('cur_model_copy'):
        os.makedirs('cur_model_copy')
    agent.save_model('cur_model_copy')
    first_old = QAgent()
    first_old.load_model('cur_model_copy')
    old_agents = [agent, first_old]

    # Training starts
    best_winning_ratio = -1
    for epoch in range(1, num_epochs + 1):

        # picking an agent from the past self
        old_num = np.random.randint(1,len(old_agents))
        agents= [agent, old_agents[len(old_agents)-old_num]]


        print(chr(27) + '[2J')
        gv.printProgressBar(epoch, num_epochs,
                            prefix = f'Agent {old_num} old\nEpoch: {epoch}',
                            length= 50)

        # During the playing of the episode the agent learn
        play_episode(game, agents)

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

            std_cur = gv.eval_visua_for_self_play(average_points_hist,
                             FLAGS,
                             victory_rates_hist,
                             evaluation_dir,
                             epoch)
            # Storing std
            std_hist.append(std_cur)


            for ag in agents:
                ag.restore_epsilon()



            # After the evaluation we add the agent to the old agents
            agent.save_model('cur_model_copy')
            new_old_agent = QAgent()
            new_old_agent.load_model('cur_model_copy')
            old_agents.append(new_old_agent)

            # Eliminating the oldest agent if maximum number of agents
            if len(old_agents) > FLAGS.max_old_agents:
                old_agents.pop(0)

        # Evaluation against random agent
        if epoch % random_every == 0:
            agents = [agent, RandomAgent()]
            for ag in agents:
                ag.make_greedy()

            winners, points = evaluate(game, agents, FLAGS.num_evaluations)
            stats_plotter(agents, points, winners, evaluation_dir,'againstRandom',epoch)

            for ag in agents:
                ag.restore_epsilon()

            # Saving the model if the agents permorm better against random agent
            if victory_rates[0] > best_winning_ratio:
                best_winning_ratio = victory_rates[0]
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
























































