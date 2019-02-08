import tensorflow as tf
import argparse
import numpy as np
import os
import random
import shutil


## our stuff import
import graphic_visualizations as gv
import environment as brisc
from evaluate import evaluate

from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.ai_agent import AIAgent
from utils import BriscolaLogger
from utils import CardsEncoding, CardsOrder, NetworkTypes, PlayerState


### New arena self play mode


class CopyAgent(QAgent):
    '''Copied agent. Identical to a QAgent, but does not update itself'''
    def __init__(self, agent):

        # create a default QAgent
        super().__init__(network=agent.network)

        # make the CopyAgent always greedy
        self.epsilon = 1.0

        # TODO: find a better way for copying the agent without saving the model
        # initialize the CopyAgent with the same weights as the passed QAgent
        if type(agent) is not QAgent:
            raise TypeError("CopyAgent __init__ requires argument of type QAgent")

        # create a temp directory where to save agent current model
        if not os.path.isdir('__tmp_model_dir__'):
            os.makedirs('__tmp_model_dir__')

        agent.save_model('__tmp_model_dir__')
        super().load_model('__tmp_model_dir__')

        # remove the temp directory after loading the model into the CopyAgent
        shutil.rmtree('__tmp_model_dir__')
        
        self.name = "CopyAgent"
        
        # A copy agent must always be greedy since it is not learning
        self.make_greedy()
        

    def update(self, *args):
        pass


def self_train(game, agent1, agent2, num_epochs, evaluate_every, num_evaluations, copy_every, model_dir = "", evaluation_dir = "evaluation_dir"):

    # initialize the list of old agents with a copy of the non trained agent
    old_agents = [[CopyAgent(agent1)], [CopyAgent(agent2)]]

    # Training starts
    best_total_wins = -1
    for epoch in range(1, num_epochs + 1):
        gv.printProgressBar(epoch, num_epochs,
                            prefix = f'Epoch: {epoch}',
                            length= 50)

        for a in [agent1,agent2]:
            other = 0 if a == agent2 else 1
            
            # picking an agent from the past as adversary
            agents = [a, random.choice(old_agents[other])]
    
            # Play a briscola game to train the agent
            brisc.play_episode(game, agents)
    
    
    
        # Evaluation step
        if epoch % evaluate_every == 0:

            # Evaluation visualization directory
            if not os.path.isdir(evaluation_dir):
                os.mkdir(evaluation_dir)

            # Greedy for evaluation
            for ag in [agent1,agent2]:
                ag.make_greedy()

            # Evaluation of the two agents
            agents = [agent1,agent2]
            winners, points = evaluate(game, agents, num_evaluations)
            gv.evaluate_summary(winners, points, agents, evaluation_dir+
                f'/epoch:{epoch} {agents[0].name}1 vs {agents[1].name}2')              
            victory_history_1v2.append(winners)
            points_history_1v2.append(points)
            
            # Evaluation against random agent
            agents = [agent1,RandomAgent()]
            winners, points = evaluate(game, agents, num_evaluations)
            gv.evaluate_summary(winners, points, agents, evaluation_dir+
                f'/epoch:{epoch} {agents[0].name}1 vs {agents[1].name}')              
            victory_history_1vR.append(winners)
            points_history_1vR.append(points)
            
            agents = [agent2,RandomAgent()]
            winners, points = evaluate(game, agents, num_evaluations)
            gv.evaluate_summary(winners, points, agents, evaluation_dir+
                f'/epoch:{epoch} {agents[0].name}2 vs {agents[1].name}')              
            victory_history_2vR.append(winners)
            points_history_2vR.append(points)                
            
            # Getting ready for more training
            for ag in [agent1,agent2]:
                ag.restore_epsilon()

            # Saving the model if the agent performs better against random agent
#            if winners[0] > best_total_wins:
#                best_total_wins = winners[0]
#                a.save_model(model_dir)

                
        if epoch % copy_every == 0:
            
            old_agents[other].append(CopyAgent(a))

            # Eliminating the oldest agent if maximum number of agents
            if len(old_agents) > FLAGS.max_old_agents:
                old_agents.pop(0)                
                
                

    return best_total_wins



def main(argv=None):

    global victory_history_1v2
    victory_history_1v2  = []
    
    global victory_history_1vR
    victory_history_1vR  = []
    
    global victory_history_2vR
    victory_history_2vR  = []


    global points_history_1v2
    points_history_1v2  = []
    
    global points_history_1vR
    points_history_1vR  = []
    
    global points_history_2vR
    points_history_2vR  = []




    # Initializing the environment
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = brisc.BriscolaGame(2, logger)

    # Initialize agent
    agent1 = QAgent(        
        FLAGS.epsilon,
        FLAGS.epsilon_increment,
        FLAGS.epsilon_max,
        FLAGS.discount,
        FLAGS.network,
        FLAGS.layers,
        FLAGS.learning_rate,
        FLAGS.replace_target_iter,
        FLAGS.batch_size
     )
    agent2 = QAgent(        
        FLAGS.epsilon,
        FLAGS.epsilon_increment,
        FLAGS.epsilon_max,
        FLAGS.discount,
        FLAGS.network,
        FLAGS.layers,
        FLAGS.learning_rate,
        FLAGS.replace_target_iter,
        FLAGS.batch_size
    )

    # Training
    best_total_wins = self_train(game, agent1, agent2,
                                    FLAGS.num_epochs,
                                    FLAGS.evaluate_every,
                                    FLAGS.num_evaluations,
                                    FLAGS.copy_every,
                                    FLAGS.model_dir)
    print('Best winning ratio : {:.2%}'.format(best_total_wins/FLAGS.num_evaluations))
    
    
    # Summary graphs
    x = [FLAGS.evaluate_every*i for i in range(1,1+len(victory_history_1v2))]

    # 1v2
    vict_hist = victory_history_1v2
    point_hist = points_history_1v2
    labels = [agent1.name+'1', agent2.name+'2']
    gv.training_summary(x, vict_hist, point_hist, labels, FLAGS, "evaluation_dir/1v2")
    
    # 1vRandom
    vict_hist = victory_history_1vR
    point_hist = points_history_1vR
    labels = [agent1.name+'1', RandomAgent().name]
    gv.training_summary(x, vict_hist, point_hist, labels, FLAGS, "evaluation_dir/1vR")
    
    # 2vRandom
    vict_hist = victory_history_2vR
    point_hist = points_history_2vR
    labels = [agent2.name+'2', RandomAgent().name]
    gv.training_summary(x, vict_hist, point_hist, labels, FLAGS, "evaluation_dir/2vR")
    
    
     # Evaluation against ai agent
    agents = [agent1,AIAgent()]
    winners, points = evaluate(game, agents, FLAGS.num_evaluations)
    gv.evaluate_summary(winners, points, agents, 'evaluation_dir'+
        f'/{agents[0].name}1 vs {agents[1].name}')              
    
    agents = [agent2,AIAgent()]
    winners, points = evaluate(game, agents, FLAGS.num_evaluations)
    gv.evaluate_summary(winners, points, agents, 'evaluation_dir'+
        f'/{agents[0].name}2 vs {agents[1].name}')            






if __name__ == '__main__':

    # Parameters
    # ==================================================

    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--model_dir", default="saved_model", help="Where to save the trained model, checkpoints and stats", type=str)
    parser.add_argument("--num_epochs", default=50000, help="Number of training games played", type=int)
    parser.add_argument("--max_old_agents", default=50, help="Maximum number of old copies of QAgent stored", type=int)

    # Evaluation parameters
    parser.add_argument("--evaluate_every", default=5000, help="Evaluate model after this many epochs", type=int)
    parser.add_argument("--num_evaluations", default=500, help="Number of evaluation games against each type of opponent for each test", type=int)

    # State parameters
    parser.add_argument("--cards_order", default=CardsOrder.APPEND, choices=[CardsOrder.APPEND, CardsOrder.REPLACE, CardsOrder.VALUE], help="Where a drawn card is put in the hand")
    parser.add_argument("--cards_encoding", default=CardsEncoding.HOT_ON_NUM_SEED, choices=[CardsEncoding.HOT_ON_DECK, CardsEncoding.HOT_ON_NUM_SEED], help="How to encode cards")
    parser.add_argument("--player_state", default=PlayerState.HAND_PLAYED_BRISCOLA, choices=[PlayerState.HAND_PLAYED_BRISCOLA, PlayerState.HAND_PLAYED_BRISCOLASEED, PlayerState.HAND_PLAYED_BRISCOLA_HISTORY], help="Which cards to encode in the player state")

    # Reinforcement Learning parameters
    parser.add_argument("--epsilon", default=0, help="How likely is the agent to choose the best reward action over a random one", type=float)
    parser.add_argument("--epsilon_increment", default=5e-5, help="How much epsilon is increased after each action taken up to epsilon_max", type=float)
    parser.add_argument("--epsilon_max", default=0.85, help="The maximum value for the incremented epsilon", type=float)
    parser.add_argument("--discount", default=0.85, help="How much a reward is discounted after each step", type=float)
    parser.add_argument("--copy_every", default=500, help="Add the copy after tot number of epochs", type=int)

    # Network parameters
    parser.add_argument("--network", default=NetworkTypes.DQN, choices=[NetworkTypes.DQN, NetworkTypes.DRQN], help="Neural Network used for approximating value function")
    parser.add_argument('--layers', default=[256, 128], help="Definition of layers for the chosen network", type=int, nargs='+')
    parser.add_argument("--learning_rate", default=1e-4, help="Learning rate for the network updates", type=float)
    parser.add_argument("--replace_target_iter", default=2000, help="Number of update steps before copying evaluation weights into target network", type=int)
    parser.add_argument("--batch_size", default=100, help="Training batch size", type=int)


    FLAGS = parser.parse_args()

    main()
    #tf.app.run()
