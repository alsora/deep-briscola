# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
# 
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import time, threading, os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras import backend as K


class Brain:
    train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self, loss_v, loss_entropy, learning_rate, min_batch, gamma_n, none_state):
        
        self.num_actions = 3
        self.num_states = 70
        self.loss_v = loss_v
        self.loss_entropy = loss_entropy
        self.learning_rate = learning_rate
        self.min_batch = min_batch
        self.gamma_n = gamma_n    
        self.none_state = none_state
        
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()	# avoid modifications

    def _build_model(self):

        l_input = Input( batch_shape=(None, self.num_states) )
        l_dense = Dense(16, activation='relu')(l_input)

        out_actions = Dense(self.num_actions, activation='softmax')(l_dense)
        out_value   = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()	# have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, self.num_states))
        a_t = tf.placeholder(tf.float32, shape=(None, self.num_actions))
        r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward
		
        p, v = model(s_t)

        log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keepdims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
        loss_value  = self.loss_v * tf.square(advantage)												# minimize value error
        entropy = self.loss_entropy * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keepdims=True)	# maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < self.min_batch:
            time.sleep(0)	# yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < self.min_batch:	# more thread could have passed without lock
                return 									# we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5*self.min_batch: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + self.gamma_n * v * s_mask	# set v to 0 where s_ is terminal state
		
        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(self.none_state)
                self.train_queue[4].append(0.)
            else:	
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s.reshape(1,70))
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s.reshape(1,70))		
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s.reshape(1,70))		
            return v


    def save_model(self, output_dir):
        '''Save the network graph and weights to disk'''
        if not output_dir:
            raise ValueError('You have to specify a valid output directory for DeepAgent.save_model')

        if not os.path.exists(output_dir):
            # if provided output_dir does not already exists, create it
            os.mkdir(output_dir)

        self.save(self.session, "./" + output_dir + '/')


    def load_model(self, saved_model_dir):
        '''Initialize a new tensorflow session loading network and weights from a saved model'''
        self.saver.restore(self.session, "./" + saved_model_dir + '/')





































