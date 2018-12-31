import numpy as np
import tensorflow as tf

import time, random, threading

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
from agents.ac3_agent import Agent






def evaluate(game, agents, num_evaluations):

    total_wins = [0] * len(agents)
    points_history = [ [] for i in range(len(agents))]

    for _ in range(num_evaluations):

        game.reset()
        keep_playing = True

        
        # capturing the state of the game
        s=np.zeros(num_state)
        players_order = game.get_players_order()
        for player_id in players_order:
            agent = agents[player_id]
            if agent.name == 'AC3':
                player = game.players[player_id]
        # add hand to s
        for i, card in enumerate(player.hand):
            number_index = i * 14 + card.number
            s[number_index] = 1
            seed_index = i * 14 + 10 + card.seed
            s[seed_index] = 1
        # add played cards to state
        for i, card in enumerate(game.played_cards):
            number_index = (i + 3) * 14 + card.number
            s[number_index] = 1
            seed_index = (i + 3) * 14 + 10 + card.seed
            s[seed_index] = 1
        # add briscola to state
        number_index = 4 * 14 + game.briscola.number
        s[number_index] = 1
        seed_index = 4 * 14 + 10 + game.briscola.seed
        s[seed_index] = 1



        while keep_playing:
            

            players_order = game.get_players_order()
            for player_id in players_order:
                
                player = game.players[player_id]
                agent = agents[player_id]
                
                if agent.name != 'AC3':
                    agent.observe(game, player, game.deck)
                    available_actions = game.get_player_actions(player_id)
                    a = agent.select_action(available_actions)
        
                else:
                    global frames

                    prob, frames = agent.act(s,frames)
                    available_actions = game.get_player_actions(player_id)
                    prob = [prob[i] for i in available_actions]
                    if sum(prob) != 0:
                        prob /= sum(prob)
                        a = np.random.choice(available_actions,p=prob)
                    else:
                        a = np.random.choice(available_actions)
        
                game.play_step(a, player_id)
                
                
                # capturing the state of the game
                s=np.zeros(num_state)
                players_order = game.get_players_order()
                for player_id in players_order:
                    agent = agents[player_id]
                    if agent.name == 'AC3':
                        player = game.players[player_id]
                # add hand to s
                for i, card in enumerate(player.hand):
                    number_index = i * 14 + card.number
                    s[number_index] = 1
                    seed_index = i * 14 + 10 + card.seed
                    s[seed_index] = 1
                # add played cards to state
                for i, card in enumerate(game.played_cards):
                    number_index = (i + 3) * 14 + card.number
                    s[number_index] = 1
                    seed_index = (i + 3) * 14 + 10 + card.seed
                    s[seed_index] = 1
                # add briscola to state
                number_index = 4 * 14 + game.briscola.number
                s[number_index] = 1
                seed_index = 4 * 14 + 10 + game.briscola.seed
                s[seed_index] = 1                




            winner_player_id, points = game.evaluate_step()

            keep_playing = game.draw_step()

        game_winner_id, winner_points = game.end_game()

        for player in game.players:
            points_history[player.id].append(player.points)
            if player.id == game_winner_id:
                total_wins[player.id] += 1

    return total_wins, points_history






class Environment(threading.Thread):
    stop_signal = False
    global num_state

    def __init__(self, eps_start, eps_end, eps_steps, thread_delay, num_actions, gamma, gamma_n, n_step_return):
        
        
        self.winner_id_hist  = []
        self.points_hist = []       
        
        self.thread_delay = thread_delay
        self.num_actions = num_actions
        
        threading.Thread.__init__(self)

        self.env = brisc.BriscolaGame(2, verbosity=brisc.LoggerLevels.TRAIN)
        
        self.agents = []
        r_agent = RandomAgent(); self.agents.append(r_agent);
        global brain
        self.agent = Agent(brain, 
                      eps_start, 
                      eps_end, 
                      eps_steps, 
                      num_actions, 
                      gamma, 
                      gamma_n, 
                      n_step_return); self.agents.append(self.agent);
        

    def runEpisode(self):
        self.env.reset()

        # capturing the state of the game
        s=np.zeros(num_state)
        players_order = self.env.get_players_order()
        for player_id in players_order:
            agent = self.agents[player_id]
            if agent.name == 'AC3':
                player = self.env.players[player_id]
        # add hand to s
        for i, card in enumerate(player.hand):
            number_index = i * 14 + card.number

            s[number_index] = 1
            seed_index = i * 14 + 10 + card.seed
            s[seed_index] = 1
        # add played cards to state
        for i, card in enumerate(self.env.played_cards):
            number_index = (i + 3) * 14 + card.number
            s[number_index] = 1
            seed_index = (i + 3) * 14 + 10 + card.seed
            s[seed_index] = 1
        # add briscola to state
        number_index = 4 * 14 + self.env.briscola.number
        s[number_index] = 1
        seed_index = 4 * 14 + 10 + self.env.briscola.seed
        s[seed_index] = 1


        R = 0
        
        
        keep_playing = True
        while keep_playing:    
            time.sleep(self.thread_delay) # yield 



            # action step 
            # TODO : refactor this code, too messy
            players_order = self.env.get_players_order()
            for player_id in players_order:
                
                player = self.env.players[player_id]
                agent = self.agents[player_id]
                
                if agent.name != 'AC3':
                    agent.observe(self.env, player, self.env.deck)
                    available_actions = self.env.get_player_actions(player_id)
                    a = agent.select_action(available_actions)
        
                else:
                    global frames

                    prob, frames = agent.act(s,frames)
                    available_actions = self.env.get_player_actions(player_id)
                    prob = [prob[i] for i in available_actions]
                    if sum(prob) != 0:
                        prob /= sum(prob)
                        a = np.random.choice(available_actions,p=prob)
                    else:
                        a = np.random.choice(available_actions)
        
                self.env.play_step(a, player_id)

            # the last player is the agent so the varibles stored are its 


            # capturing the next state
            
            s_=np.zeros(num_state)
            # add hand to s
            
            players_order = self.env.get_players_order()
            for player_id in players_order:
                agent = self.agents[player_id]
                if agent.name == 'AC3':
                    player = self.env.players[player_id]

            for i, card in enumerate(player.hand):
                number_index = i * 14 + card.number
                s_[number_index] = 1
                seed_index = i * 14 + 10 + card.seed
                s_[seed_index] = 1
            # add played cards to state
            for i, card in enumerate(self.env.played_cards):
                number_index = (i + 3) * 14 + card.number
                s_[number_index] = 1
                seed_index = (i + 3) * 14 + 10 + card.seed
                s_[seed_index] = 1
            # add briscola to state
            number_index = 4 * 14 + self.env.briscola.number
            s_[number_index] = 1
            seed_index = 4 * 14 + 10 + self.env.briscola.seed
            s_[seed_index] = 1




            players_order = self.env.get_players_order()
            i = 0
            for player_id in players_order:
                agent = self.agents[player_id]
                if agent.name == 'AC3':
                    break
                i += 1
            r = self.env.get_rewards_from_step()[i]

            keep_playing = self.env.draw_step()

            if not keep_playing: # terminal state
                s_ = None
                r = 2 if r > 0 else -2
                winner_player_id, winner_points = self.env.end_game()

            self.agent.train(s, a, r, s_)

            s = s_
            R += r
            

        return R

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        
    #   TODO : adding make_greedy function to ac3 agent
#        for ag in agents:
#            ag.make_greedy()
    
    
        # TODO : adding num_evaluation correctly
        victory_rates, average_points = evaluate(self.env, self.agents, 150)
    
    
        
        
        self.stop_signal = True
        return victory_rates, average_points





#def train(game, agents, num_epochs, evaluate_every, num_evaluations, model_dir = ""):
#
#    best_winning_ratio = -1
#    for epoch in range(1, num_epochs + 1):
#        print ("Epoch: ", epoch, end='\r')
#
#        game_winner_id, winner_points = play_episode(game, agents)
#
#    return best_winning_ratio


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



frames = 0
num_state = 70


def main(argv=None):
    
    brain = AC3_Brain(3, num_state, FLAGS.loss_v, FLAGS.loss_entropy, FLAGS.learning_rate, FLAGS.min_batch, FLAGS.gamma_n, np.zeros(num_state))

    envs = [Environment(FLAGS.eps_start,
                    FLAGS.eps_stop,
                    FLAGS.eps_step,
                    FLAGS.thread_delay,
                    3,
                    FLAGS.gamma, 
                    FLAGS.gamma_n, 
                    FLAGS.n_step_return) for _ in range(FLAGS.threads)]
    opts = [Optimizer() for i in range(FLAGS.optimizers)]
    
    for o in opts:
    	o.start()
    for e in envs:
    	e.start()
    
    # run time
    time.sleep(FLAGS.run_time)
    
    data = []
    
    for e in envs:
    	data.append(e.stop())
    for e in envs:
    	e.join()
    
    for o in opts:
    	o.stop()
    for o in opts:
    	o.join()
    
    print("Training finished")
    
    
    
    
evaluate(envs[0].env, envs[0].agents, 100)
    
    
    


if __name__ == '__main__':

    # Parameters
    # ==================================================

    # Model directory
    tf.flags.DEFINE_string("model_dir", "saved_model", "Where to save the trained model, checkpoints and stats (default: pwd/saved_model)")

    # Training parameters
    tf.flags.DEFINE_integer("run_time", 30, '')
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















