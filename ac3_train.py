import numpy as np
import tensorflow as tf

import time, random, threading, os

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K


## our stuff imports
import graphic_visualizations as gv
import environment as brisc
from graphic_visualizations import stats_plotter

from networks.ac3 import Brain as AC3_Brain

from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.ai_agent import AIAgent
from agents.ac3_agent import AgentAC3



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



# =============================================================================
# ENVIROMENT
# =============================================================================

class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, eps_start, eps_end, eps_steps, thread_delay, gamma, gamma_n, n_step_return):
        
        # Initialize the threading
        threading.Thread.__init__(self)
        self.thread_delay = thread_delay

        # Initialize the game
        self.game = brisc.BriscolaGame(2, verbosity=brisc.LoggerLevels.TRAIN)
        
        self.epoch = 0
        
        # Initialize statistics
        self.victory_rates_hist = []
        self.average_points_hist = []
        self.std_hist = []
        
        # Initialize the agents
        # TODO : give the possibility to choose the opponent
        self.agents = []
        r_agent = RandomAgent(); self.agents.append(r_agent);
        global brain
        self.agent = AgentAC3(brain, 
                      eps_start, 
                      eps_end, 
                      eps_steps, 
                      gamma, 
                      gamma_n, 
                      n_step_return); self.agents.append(self.agent);
        

    def runEpisode(self):
        self.game.reset()

        winner_player_id, winner_points = play_episode(self.game, self.agents)
        self.epoch += 1
        
            # EVALUATION VISUALISATION
            # TODO : there a bug during the plotting 
#        if self.epoch % 100 == 0:
#            for ag in self.agents:
#                ag.make_greedy()
#            victory_rates, average_points = evaluate(self.game, self.agents, 100)#num_evaluation
#            self.victory_rates_hist.append(victory_rates)
#            self.average_points_hist.append(average_points)
#            
#            if not os.path.isdir("evaluation_dir"):
#                os.mkdir("evaluation_dir")
#            
#            std_cur = gv.eval_visua_for_self_play(self.average_points_hist,
#                             FLAGS,
#                             self.victory_rates_hist,
#                             "evaluation_dir",
#                             self.epoch)
#            # Storing std
#            self.std_hist.append(std_cur)
#            for ag in self.agents:
#                ag.restore_epsilon()  
#                
                
        R = sum(self.game.rewards_hist[1])
        return R

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True


#---------
class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True




def main(argv=None):
    
    brain = AC3_Brain(FLAGS.loss_v, FLAGS.loss_entropy, FLAGS.learning_rate, FLAGS.min_batch, FLAGS.gamma_n, np.zeros(70))

    # TRAINING
    envs = [Environment(FLAGS.eps_start,
                    FLAGS.eps_stop,
                    FLAGS.eps_step,
                    FLAGS.thread_delay,
                    FLAGS.gamma, 
                    FLAGS.gamma_n, 
                    FLAGS.n_step_return) for _ in range(FLAGS.threads)]
    opts = [Optimizer() for i in range(FLAGS.optimizers)]
    
    for o in opts:
    	o.start()
    for e in envs:
    	e.start()
    
    # run time
    #time.sleep(FLAGS.run_time)
    
    input("insert something to stop:")
    
    for e in envs:
    	e.stop()
    for e in envs:
    	e.join()
    
    for o in opts:
    	o.stop()
    for o in opts:
    	o.join()
    
    print("Training finished")
    

    # TEST
    
    # TODO : add a functioning save_model function
    # brain.save_model(FLAGS.model_dir)

    e = envs[1]
    print(e.epoch)
    
    e.agents[1].make_greedy()            
        
    
    winners, points = evaluate(e.game, [e.agents[1], AIAgent()], 1000)
    stats_plotter(e.agents, points, winners, 'evaluation_dir' ,'final test against ai','')
    
    winners, points = evaluate(e.game, [e.agents[1], RandomAgent()], 1000)
    stats_plotter(e.agents, points, winners, 'evaluation_dir' ,'final test against Random','')

    for ag in e.agents:
        ag.restore_epsilon()
        
    


if __name__ == '__main__':

    # Parameters
    # ==================================================

    # Model directory
    tf.flags.DEFINE_string("model_dir", "saved_model", "Where to save the trained model, checkpoints and stats (default: pwd/saved_model)")

    # Training parameters
    tf.flags.DEFINE_integer("run_time", 3, '')
    tf.flags.DEFINE_integer("threads", 8, '')
    tf.flags.DEFINE_integer("optimizers", 2, '')
    tf.flags.DEFINE_integer("n_step_return", 8, '')
    tf.flags.DEFINE_integer("min_batch", 32, "Batch Size")
    tf.flags.DEFINE_integer("num_epochs", 32, "Number of training epochs")
    
    tf.flags.DEFINE_float("thread_delay", .001, '')
    tf.flags.DEFINE_float("gamma", .99, '')
    tf.flags.DEFINE_float("loss_v", .5, '')
    tf.flags.DEFINE_float("loss_entropy", .01, '')
    tf.flags.DEFINE_float("gamma_n", tf.flags.FLAGS.gamma ** tf.flags.FLAGS.n_step_return , '')
    tf.flags.DEFINE_float("eps_start", 0.4, "How likely is the agent to choose the best reward action over a random one (default: 0)")
    tf.flags.DEFINE_float("eps_step", 75000., "How much epsilon is increased after each action taken up to epsilon_max (default: 5e-6)")
    tf.flags.DEFINE_float("eps_stop", .15, "The maximum value for the incremented epsilon (default: 0.85)")
    tf.flags.DEFINE_float("learning_rate", 5e-3, "The learning rate for the network updates (default: 1e-4)")


    # Evaluation parameters
    tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model after this many steps (default: 1000)")
    tf.flags.DEFINE_integer("num_evaluations", 500, "Evaluate on these many episodes for each test (default: 500)")

    FLAGS = tf.flags.FLAGS

    tf.app.run()















