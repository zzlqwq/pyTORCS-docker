import numpy as np
import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Dense, Input, Multiply, BatchNormalization, Activation
from keras.initializers import RandomNormal
from keras.layers import concatenate

from keras.optimizers import Adam

from torcs_client.utils import SimpleLogger as log

"""
Actor network:
stochastic funcion approssimator for the deterministic policy map u : S -> A
(with S set of states, A set of actions)
"""


class Actor(object):
    def __init__(self, state_dims, action_dims, lr, batch_size, tau,
                 fcl1_size, fcl2_size, upper_bound, save_dir):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.lr = lr
        # learning rate
        self.batch_size = batch_size
        self.tau = tau
        # polyak averaging speed
        self.fcl1_size = fcl1_size
        self.fcl2_size = fcl2_size
        self.upper_bound = upper_bound

        # load model if present
        load_model_flag = False
        if load_model_flag:
            self.model = tf.keras.models.load_model(save_dir + "/actor")
            self.target_model = tf.keras.models.load_model(save_dir + "/actor_target")
            log.info("Loaded saved actor models")
        else:
            self.model = self.build_network()
            # duplicate model for target
            self.target_model = self.build_network()
            self.target_model.set_weights(self.model.get_weights())
            self.model.summary()

        self.optimizer = Adam(self.lr)

    def build_network(self):
        """
        Builds the model. Consists of two fully connected layers with batch norm.
        """
        # -- input layer --
        input_layer = Input(shape=self.state_dims)
        # -- first fully connected layer --
        h0 = Dense(self.fcl1_size, activation='relu')(input_layer)
        # -- second fully connected layer --
        h1 = Dense(self.fcl2_size, activation='relu')(h0)
        steering = Dense(1, activation='tanh', kernel_initializer=RandomNormal(0, 1e-4),
                         bias_initializer="zeros")(h1)
        acceleration = Dense(1, activation='sigmoid', kernel_initializer=RandomNormal(0, 1e-4),
                             bias_initializer="zeros")(h1)
        brake = Dense(1, activation='sigmoid', kernel_initializer=RandomNormal(0, 1e-4),
                      bias_initializer="zeros")(h1)
        # -- output layer --
        output_layer = concatenate([steering, acceleration, brake])

        model = Model(input_layer, output_layer)
        return model

    @tf.function
    def train(self, states, critic_model):
        """
        Update the weights with the new critic evaluation
        """
        with tf.GradientTape() as tape:
            actions = self.model(states, training=True)
            q_value = critic_model([states, actions], training=True)
            loss = -tf.math.reduce_mean(q_value)
        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        return loss

    def update_target(self):
        """
        Update the target weights using tau as speed. The tracking function is
        defined as:
        target = tau * weights + (1 - tau) * target
        """
        # faster updates woth graph mode
        self._transfer(self.model.variables, self.target_model.variables)

    @tf.function
    def _transfer(self, model_weights, target_weights):
        for (weight, target) in zip(model_weights, target_weights):
            # update the target values
            target.assign(weight * self.tau + target * (1 - self.tau))
