import numpy as np

import random


class AgentAC3:
    
    def __init__(self, brain, eps_start, eps_end, eps_steps, gamma, gamma_n, n_step_return):
        
        self.name = 'AC3'
        
        self.frames = 0
        self.n_features = 70        
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_steps = eps_steps
        self.num_actions = 3
        self.gamma = gamma
        self.gamma_n = gamma_n
        self.n_step_return = n_step_return
        
        self.brain = brain
        self.greedy = False
        
        self.state = np.zeros(self.n_features)
        self.memory = []	# used for n_step return
        self.R = 0.
        
    def observe(self, game, player, deck):
        
        state = np.zeros(self.n_features)
        # add hand to state
        for i, card in enumerate(player.hand):
            number_index = i * 14 + card.number
            state[number_index] = 1
            seed_index = i * 14 + 10 + card.seed
            state[seed_index] = 1
        # add played cards to state
        for i, card in enumerate(game.played_cards):
            number_index = (i + 3) * 14 + card.number
            state[number_index] = 1
            seed_index = (i + 3) * 14 + 10 + card.seed
            state[seed_index] = 1
        # add briscola to state
        number_index = 4 * 14 + game.briscola.number
        state[number_index] = 1
        seed_index = 4 * 14 + 10 + game.briscola.seed
        state[seed_index] = 1
        # add seen cards
        #for card in game.history:
            #card_index = 5 * 14 + card.id
            #state[card_index] = 1

        self.last_state = self.state
        self.state = state
        
        
    def select_action(self, available_actions):

        if self.state is None:
            raise ValueError("DeepAgent.select_action called before observing the state")

        eps = self.getEpsilon()			
        self.frames += 1


        if random.random() < eps and not self.greedy:
            action = np.random.choice(available_actions)
        else:
            # getting the probabilities
            p = self.act(self.state)
            # setting to zero the probability for not available actions
            p = [p[i] if i in available_actions else 0 for i in range(self.num_actions) ]
            # resetting the sum to 1
            pr = [i/sum(p) for i in p]

            # if the probabilities for the available action is zero then random
            if sum(p) == 0:
                action = np.random.choice(available_actions)
            else:
                action = np.random.choice(self.num_actions,p=pr)

        # store the chosen action
        self.action = action
        return action       
    
    def update(self, reward):
        ''' After receiving a reward the agent has all collected [s, a, r, s_]'''

        '''
        if self.wrong_move:
            # reduce the reward if the agent's last move has been a not allowed move
            self.reward = -10
            self.wrong_move = False
        else:
            self.reward = reward
        '''
        # update last reward
        self.reward = reward

        self.train(self.last_state, self.action, self.reward, self.state)
        
        
    def make_greedy(self):
        self.greedy = True 

    def restore_epsilon(self):
        self.greedy = False 

        
    def getEpsilon(self):
        if(self.frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + self.frames * (self.eps_end - self.eps_start) / self.eps_steps	# linearly interpolate

    def act(self, s):
        p = self.brain.predict_p(s)[0]

        return p
    
	
    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _  = memory[0]
            _, _, _, s_ = memory[n-1]

        return s, a, self.R, s_

        a_cats = np.zeros(self.num_actions)	# turn action into one-hot representation
        a_cats[a] = 1 

        self.memory.append( (s, a_cats, r, s_) )

        self.R = ( self.R + r * self.gamma_n ) / self.gamma

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)

                self.R = ( self.R - self.memory[0][2] ) / self.gamma
                self.memory.pop(0)		

            self.R = 0

        if len(self.memory) >= self.n_step_return:
            s, a, r, s_ = get_sample(self.memory, self.n_step_return)
            self.brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)	
	
	# possible edge case - if an episode ends in <N steps, the computation is incorrect
