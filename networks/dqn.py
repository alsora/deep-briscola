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


class DQN(BaseNetwork):

    def __init__(self, n_actions, n_features, layers=[256, 128], learning_rate=1e-3, batch_size=100, replace_target_iter=2000, discount=0.85):
        # initialize base class
        super().__init__()

        # network parameters
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = discount
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter

        # update parameters
        self.learn_iter = 0
        self.update_each = 1
        self.update_after = 5000

        # layers parameters
        self.layers = layers

        # init vars
        self.learn_step_counter = 0
        self.wrong_move = False
        self.session = None

        # create replay memroy
        capacity = 10000
        self.replay_memory = ReplayMemory(capacity, self.n_features)

        # create network
        self.create_network()
        self.initialize_session()


    def create_network(self):

        with self.graph.as_default():

            # input placeholders
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='states')  # input State
            self.a = tf.placeholder(tf.int32, [None, ], name='actions')  # input Action
            self.r = tf.placeholder(tf.float32, [None, ], name='rewards')  # input Reward
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='states_')  # input Next State
            self.terminal = tf.placeholder(tf.float32, [None, ], name='terminal') # indication if next state is terminal

            w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            # evaluation network
            with tf.variable_scope('eval_net'):
                last_tensor = self.s
                for layer_size in self.layers:
                    last_tensor = tf.layers.dense(last_tensor, layer_size, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer)

                self.q = tf.layers.dense(last_tensor, self.n_actions, kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer, name='q')

            # target network
            with tf.variable_scope('target_net'):
                last_tensor = self.s_
                for layer_size in self.layers:
                    last_tensor = tf.layers.dense(last_tensor, layer_size, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer)

                self.q_next = tf.layers.dense(last_tensor, self.n_actions, kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer, name='q_next')

            with tf.variable_scope('predictions'):
                # predicted actions according to evaluation network
                self.argmax_action = tf.argmax(self.q, 1, output_type=tf.int32, name="argmax")
            with tf.variable_scope('q_target'):
                # discounted reward on the target network
                q_target = self.r + (1. - self.terminal) * self.gamma * tf.reduce_max(self.q_next, axis=1, name='q_target')
                # stop gradient to avoid updating target network
                self.q_target = tf.stop_gradient(q_target)
            with tf.variable_scope('q_wrt_a'):
                # q value of chosen action
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
                self.q_wrt_a = tf.gather_nd(params=self.q, indices=a_indices)
            with tf.variable_scope('loss'):
                # loss computed as difference between predicted q[a] and (current_reward + discount * q_target[best_future_action])
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_wrt_a, name='td_error'))
            with tf.variable_scope('train'):
                opt = tf.train.AdamOptimizer(self.learning_rate)
                grads_and_vars = opt.compute_gradients(self.loss)
                self._train_op = opt.apply_gradients(grads_and_vars, name="optimizer")

            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

            with tf.variable_scope('hard_replacement'):
                # operator for assiging evaluation network weights to the target network
                self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]


    def get_q_table(self, state):
        ''' Compute q table for current state'''

        states_op = self.session.graph.get_operation_by_name("states").outputs[0]
        #argmax_op = self.session.graph.get_operation_by_name("predictions/argmax").outputs[0]
        q_op = self.session.graph.get_operation_by_name("eval_net/q/BiasAdd").outputs[0]

        input_state = np.expand_dims(state, axis=0)
        q = self.session.run([q_op], feed_dict={states_op: input_state})

        return q[0][0]

    def store(self, last_state, action, reward, state, terminal):
        ''' Store the current experience in memory '''

        # stacks together all states element
        state_vector = np.hstack((last_state, action, reward, state, terminal))

        self.replay_memory.push(state_vector)


    def learn(self, last_state, action, reward, state, terminal):
        ''' Sample from memory and train neural network on a batch of experiences '''

        self.store(last_state, action, reward, state, terminal)

        # check if it's time to update the network
        self.learn_iter += 1
        if self.learn_iter % self.update_each != 0 or self.learn_iter < self.update_after:
            return

        if self.replay_memory.size() < self.batch_size:
            # there are not enough samples for a training step in the replay memory
            return

        # get a batch of samples from replay memory
        batch_memory = self.replay_memory.sample(self.batch_size)

        # run a newtork training step
        _, loss = self.session.run(
            [self._train_op, self.loss,],
            feed_dict={
                self.s: batch_memory[:, : self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features-1:-1],
                self.terminal: batch_memory[:, -1],
            })

        # check if it's time to copy the target network into the evaluation network
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.session.run(self.target_replace_op)
            print("Loss: ", loss)

        self.learn_step_counter += 1


