import tensorflow as tf
import os


class BaseNetwork:

    def __init__(self):
        '''Defines self.graph as an empty tensorflow graph'''
        self.graph = tf.Graph()


    def initialize_session(self):
        '''Defines self.sess and initialize the variables'''
        session_conf = tf.ConfigProto(
            allow_soft_placement = True,
            log_device_placement = False)

        self.session = tf.Session(config = session_conf, graph=self.graph)

        with self.graph.as_default():
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        self.session.run(self.init)


    def save_model(self, output_dir):
        '''Save the network graph and weights to disk'''
        if not output_dir:
            raise ValueError('You have to specify a valid output directory for DeepAgent.save_model')

        if not os.path.exists(output_dir):
            # if provided output_dir does not already exists, create it
            os.mkdir(output_dir)

        self.saver.save(self.session, './' + output_dir + '/')


    def load_model(self, saved_model_dir):
        '''Initialize a new tensorflow graph and session loading network and weights from a saved model'''
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.train.import_meta_graph('./' + saved_model_dir + '/.meta')

        self.initialize_session()
        self.saver.restore(self.session, './' + saved_model_dir + '/')