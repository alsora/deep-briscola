import tensorflow as tf
import environment as brisc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

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



def self_train(game, agent, num_epochs, evaluate_every, num_evaluations, model_dir = "", evaluation_dir = "evaluation_dir"):

    # Initializing the list of agents with the agent and a copy of him
    if not os.path.isdir('cur_model_copy'):
        os.makedirs('cur_model_copy')
    agent.save_model('cur_model_copy')
    first_old = QAgent()
    first_old.load_model('cur_model_copy')
    old_agents = [agent, first_old]
    
    
    
    best_winning_ratio = -1
    for epoch in range(1, num_epochs + 1):
        print(chr(27) + '[2J')
        
        # picking an agent from the past self
        old_num = np.random.randint(1,len(old_agents))
        agents= [agent, old_agents[len(old_agents)-old_num]] 
        
        print (f"Epoch: {epoch} Agent {old_num} old", end='\r')
                

        game_winner_id, winn500er_points = play_episode(game, agents)

        if epoch % evaluate_every == 0:
            for agent in agents:
                agent.make_greedy()
            victory_rates, average_points = evaluate(game, agents, num_evaluations)
            victory_rates_hist.append(victory_rates)
            average_points_hist.append(average_points)
            
            # EVALUATION VISUALISATION
            if not os.path.isdir(evaluation_dir):
                os.mkdir(evaluation_dir)
            
            df  = np.array(average_points_hist)
            evaluation_num = len(df[:,0,0])
            eval_df = np.array(average_points_hist)[evaluation_num-1,:,:]
            eval_df = pd.DataFrame(eval_df.T, columns = ["Agent 0","Agent 1"])
            
            eval_df.plot()
            plt.hlines([np.mean(eval_df.values),
                        np.mean(eval_df.values)+np.std(eval_df.values),
                        np.mean(eval_df.values)-np.std(eval_df.values)],
                       0, len(df[0,0,:]), color = ['green','red','red'], 
                       label = 'mean+-std')
            plt.title(f"""Eval:{evaluation_num}_
                      epoch:{evaluation_num*FLAGS.evaluate_every}_
                      pctWinBest:{max(victory_rates_hist[-1])}_
                      std:{np.std(eval_df.values).round(2)}""".replace('\n','').replace(' ',''))
            plt.ylim(0,120)
            plt.xlabel("Evaluation step")
            plt.ylabel("Points")
            plt.legend()
            plt.savefig(f"{evaluation_dir}/fig_{epoch}")
            plt.close()
            
            # Storing std
            std_hist.append(np.std(eval_df.values).round(2))

            
            for agent in agents:
                agent.restore_epsilon()
            #print("DeepAgent wins ", "{:.2f}".format(victory_rates[0]), "% with average points ", "{:.2f}".format(average_points[0]))
            if victory_rates[0] > best_winning_ratio:
                best_winning_ratio = victory_rates[0]
                #agents[0].save_model(model_dir)
                
            # After the evaluation we add the agent to the old agents
            agent.save_model('cur_model_copy')
            new_old_agent = QAgent()
            new_old_agent.load_model('cur_model_copy')
            old_agents.append(new_old_agent)
            

    return best_winning_ratio




def main(argv=None):

    global victory_rates_hist
    victory_rates_hist  = []
    
    global average_points_hist 
    average_points_hist = []
    
    global std_hist
    std_hist = []
    
    # Initializing the environment
    game = brisc.BriscolaGame(2, verbosity=brisc.LoggerLevels.TRAIN)

    # Initialize agent
    agent = QAgent(
        FLAGS.epsilon, FLAGS.epsilon_increment, FLAGS.epsilon_max, FLAGS.discount,
        FLAGS.learning_rate)
    
    self_train(game, agent, FLAGS.num_epochs, FLAGS.evaluate_every, 
          FLAGS.num_evaluations, FLAGS.model_dir, FLAGS.evaluation_dir)


    # SUMMARY GRAPH
    df = np.vstack([np.array(victory_rates_hist).T,np.array(std_hist)]).T
    vict_rate = pd.DataFrame(df, columns = ["Agent 0 win_rate","Agent 1 win_rate", "Std"])
    
    vict_rate['Agent 0 win_rate'].plot(secondary_y=False, 
                                       color = 'lightgreen',
                                       label='Agent 0 (left)')
    vict_rate['Agent 1 win_rate'].plot(secondary_y=False, 
                                       color = 'lightblue',
                                       label='Agent 1 (left)')
    plt.hlines([np.mean(vict_rate.values[:,0]),
                np.mean(vict_rate.values[:,1])],
               0, len(vict_rate)-1, color = ['green','blue'], 
               label = 'means')
    plt.ylabel('WinRate')
    plt.legend()
    
    vict_rate.Std.plot(secondary_y=True, label="Std (right)", color = 'red', 
                       alpha = 0.8, linestyle='-.')
    plt.ylabel('StandardDeviation', rotation=270, labelpad=15)
    plt.legend()
    plt.savefig(f"{FLAGS.evaluation_dir}/last")







if __name__ == '__main__':

    # Parameters
    # ==================================================

    # Directories
    tf.flags.DEFINE_string("model_dir", "saved_model", "Where to save the trained model, checkpoints and stats (default: pwd/saved_model)")
    tf.flags.DEFINE_string("evaluation_dir", "evaluation_dir3", "Where to save the trained model, checkpoints and stats (default: pwd/saved_model)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
    tf.flags.DEFINE_integer("num_epochs", 61, "Number of training epochs")

    # Deep Agent parameters
    tf.flags.DEFINE_float("epsilon", 0, "How likely is the agent to choose the best reward action over a random one (default: 0)")
    tf.flags.DEFINE_float("epsilon_increment", 5e-5, "How much epsilon is increased after each action taken up to epsilon_max (default: 5e-6)")
    tf.flags.DEFINE_float("epsilon_max", 0.85, "The maximum value for the incremented epsilon (default: 0.85)")
    tf.flags.DEFINE_float("discount", 0.85, "How much a reward is discounted after each step (default: 0.85)")

    # Network parameters
    tf.flags.DEFINE_float("learning_rate", 1e-4, "The learning rate for the network updates (default: 1e-4)")


    # Evaluation parameters
    tf.flags.DEFINE_integer("evaluate_every", 30, "Evaluate model after this many steps (default: 1000)")
    tf.flags.DEFINE_integer("num_evaluations", 100, "Evaluate on these many episodes for each test (default: 500)")

    FLAGS = tf.flags.FLAGS

    tf.app.run()
























































