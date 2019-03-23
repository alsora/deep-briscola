import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import random

from networks.base_network import BaseNetwork

class ReplayMemory:

    def __init__(self, capacity, n_features):

        # initialize zero memory, each sample in memory has size [s + a + r + s_ + t]
        # where s and s_ are 1xn_features, a r t are scalar

        self.capacity = capacity
        self.event_size = n_features * 2 + 3
        self.memory = np.zeros((self.capacity, self.event_size))
        self.memory_counter = 0

    def push(self, item):
        # get the index where to insert the event
        index = self.memory_counter % self.capacity
        self.memory[index, :] = item

        # increment memory_counter avoiding overflow (I only need to keep track if memory is full or not)
        self.memory_counter += 1
        if self.memory_counter == (self.capacity * 2):
            self.memory_counter = self.capacity

    def sample(self, batch_size):

        if self.memory_counter > self.capacity:
            # the replay memory is all written, so I can sample on all the array size [0, self.capacity]
            sample_index = np.random.choice(self.capacity, size=batch_size)
        else:
            # only part of the replay memory is written, so I sample up to where it's written [0, self.memory_counter]
            sample_index = np.random.choice(self.memory_counter, size=batch_size)

        batch_memory = self.memory[sample_index, :]
        return batch_memory

    def size(self):
        return min(self.capacity, self.memory_counter)


class AC(BaseNetwork):

    def __init__(self, n_actions, n_features, layers=[256, 128], learning_rate=1e-3, batch_size=100, replace_target_iter=2000, discount=0.85):
        # initialize base class
        super().__init__()

        # network parameters
        self.n_features = n_features
        self.n_actions = n_actions


        self.learning_rate = 7e-4
        self.alpha = 0.99
        self.epsilon = 1e-5

        self.gamma = discount
        self.batch_size = batch_size

        # update parameters
        self.learn_step_counter = 0
        self.update_each = 1
        self.update_after = 5000

        self.max_grad_norm = 2.5
        self.reg_param = 0.001
        self.entropy_coeff = 0.01
        self.vf_coeff = 0.5

        # layers parameters
        self.layers = layers



        # create replay memroy
        capacity = 10000
        self.replay_memory = ReplayMemory(capacity, self.n_features)

        # rollout buffer
        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []

        # create network
        self.session = None
        self.create_network()
        self.initialize_session()

    def create_network(self):

        with self.graph.as_default():

            # input placeholders
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='states')  # input State
            self.a = tf.placeholder(tf.int32, [None, ], name='actions')  # input Action
            self.td_error_input = tf.placeholder(tf.float32, [None, ], "td_error_input")
            self.terminal = tf.placeholder(tf.float32, [None, ], name='terminal') # indication if next state is terminal

            w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            with tf.variable_scope('actor_network'):

                last_tensor = self.s
                for layer_size in self.layers:
                    last_tensor = tf.layers.dense(last_tensor, layer_size, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer)

                self.action_probs = tf.layers.dense(last_tensor, self.n_actions, activation=tf.nn.softmax, kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer, name='action_probs')

                with tf.variable_scope('train'):

                    # logarithmic probability of chosen action
                    a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
                    prob_of_a = tf.gather_nd(params=self.action_probs, indices=a_indices)
                    log_prob_of_a = tf.log(prob_of_a)



                    #log_prob = tf.log(self.action_probs[0, self.a])
                    self.expected_value = tf.reduce_mean(log_prob_of_a * self.td_error_input) # advantage (TD_error) guided loss

                    opt = tf.train.AdamOptimizer(self.learning_rate)
                    grads_and_vars = opt.compute_gradients(-self.expected_value) # maximize the expected value
                    self.actor_train_op = opt.apply_gradients(grads_and_vars, name="optimizer")



            self.v_next = tf.placeholder(tf.float32, [None, ], "v_next")
            self.r = tf.placeholder(tf.float32, [None, ], name='rewards')  # input Reward


            with tf.variable_scope('critic_network'):

                last_tensor = self.s
                for layer_size in self.layers:
                    last_tensor = tf.layers.dense(last_tensor, layer_size, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer)

                self.v = tf.layers.dense(last_tensor, 1, activation=None, kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer, name='V')
            
            
                with tf.variable_scope('train'):

                    reshaped_v = tf.reshape(self.v, [self.batch_size])
                    self.td_error = self.r + (1. - self.terminal) * self.gamma * self.v_next - reshaped_v  # difference between real and predicted value
                    self.squared_td_error = tf.square(self.td_error) 
                    # critic has to learn faster than actor
                    opt = tf.train.AdamOptimizer(10 * self.learning_rate)
                    grads_and_vars = opt.compute_gradients(self.squared_td_error)
                    self.critic_train_op = opt.apply_gradients(grads_and_vars, name="optimizer")


            
  
    def get_q_table(self, state):
        ''' Compute q table for current state'''

        states_op = self.session.graph.get_operation_by_name("states").outputs[0]

        input_state = np.expand_dims(state, axis=0)


        action_probs = self.session.run([self.action_probs], feed_dict={states_op: input_state})

        # subtract 1e-5 to all terms to ensure that their sum is less than 1
        probs = [x-1e-5 for x in action_probs[0][0]]

        # sample 1 item from actions with probabilities "action_probs"
        one_hot_sampled_action = np.random.multinomial(1, probs)


        return one_hot_sampled_action


    def store(self, last_state, action, reward, state, terminal):
        ''' Store the current experience in memory '''

        # stacks together all states element
        state_vector = np.hstack((last_state, action, reward, state, terminal))

        self.replay_memory.push(state_vector)


    def learn(self, last_state, action, reward, state, terminal):

        self.store(last_state, action, reward, state, terminal)

        # check if it's time to update the network
        self.learn_step_counter += 1
        if self.learn_step_counter < self.update_after:
            return

        if self.replay_memory.size() < self.batch_size:
            # there are not enough samples for a training step in the replay memory
            return

        # get a batch of samples from replay memory
        batch_memory = self.replay_memory.sample(self.batch_size)

        v_next = self.session.run([self.v], feed_dict={self.s:batch_memory[:, -self.n_features-1:-1]})

        # v_next has a strange shape 1 x N x 1 
        v_next = [x[0] for x in v_next[0]]

        # gradient = grad[r + gamma * V(s_) - V(s)]
        _, td_error, v = self.session.run(
            [self.critic_train_op, self.td_error, self.v],
            feed_dict={
                self.s: batch_memory[:, : self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.v_next: v_next,
                self.terminal: batch_memory[:, -1],
            })

        # true_gradient = grad[logPi(s,a) * td_error]
        _, exp_v = self.session.run(
            [self.actor_train_op, self.expected_value],
            feed_dict={
                self.s: batch_memory[:, : self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.td_error_input: td_error,
                self.terminal: batch_memory[:, -1],
            })
        