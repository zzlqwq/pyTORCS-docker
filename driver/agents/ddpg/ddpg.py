import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras import backend as K

import os
import time
import h5py

from agents.ddpg.utils.replay_buffer import ReplayBuffer
from agents.ddpg.utils.action_noise import OUActionNoise
from agents.ddpg.utils.OU import OU
from agents.ddpg.network.actor import Actor
from agents.ddpg.network.critic import Critic

from torcs_client.reward import LocalReward
from torcs_client.utils import SimpleLogger as log


class DDPG(object):
    """
    DDPG agent
    """

    def __init__(self, state_dims, action_dims, action_boundaries, hyperparams):

        # physical_devices = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)

        actor_lr = hyperparams["actor_lr"]
        critic_lr = hyperparams["critic_lr"]
        batch_size = hyperparams["batch_size"]
        gamma = hyperparams["gamma"]
        self.buf_size = int(hyperparams["buf_size"])
        tau = hyperparams["tau"]
        fcl1_size = hyperparams["fcl1_size"]
        fcl2_size = hyperparams["fcl2_size"]
        guided_episode = hyperparams["guided_episode"] - 1
        save_dir = hyperparams["save_dir"]

        noise_phi = hyperparams["noise_phi"]

        # action size
        self.n_states = state_dims[0]
        # state size
        self.n_actions = action_dims[0]
        self.batch_size = batch_size

        self.guided_episode = guided_episode
        self.save_dir = save_dir

        # environmental action boundaries
        self.lower_bound = action_boundaries[0]
        self.upper_bound = action_boundaries[1]

        self.noise_phi = noise_phi

        # experience replay buffer
        self._memory = ReplayBuffer(self.buf_size, input_shape=state_dims, output_shape=action_dims)
        # noise generator
        self._noise = OUActionNoise(mu=np.zeros(action_dims))
        self.OU = OU()
        # Bellman discount factor
        self.gamma = gamma

        self.prev_accel = 0

        self.track = ""

        self.epsilon = 1.0

        # turn off most logging
        logging.getLogger("tensorflow").setLevel(logging.FATAL)

        # date = datetime.now().strftime("%m%d%Y_%H%M%S")
        # path_actor = "./models/actor/actor" + date + ".h5"
        # path_critic = "./models/critic/actor" + date + ".h5"

        # actor class
        self.actor = Actor(state_dims=state_dims, action_dims=action_dims,
                           lr=actor_lr, batch_size=batch_size, tau=tau,
                           upper_bound=self.upper_bound, save_dir=self.save_dir,
                           fcl1_size=fcl1_size, fcl2_size=fcl2_size)
        # critic class
        self.critic = Critic(state_dims=state_dims, action_dims=action_dims,
                             lr=critic_lr, batch_size=batch_size, tau=tau,
                             save_dir=self.save_dir, fcl1_size=fcl1_size, fcl2_size=fcl2_size)

    def get_action(self, state, episode, track, is_training):
        """
        Return the best action in the past state, according to the model
        in training. Noise added for exploration
        """
        if self.track != track:
            self.track = track

        state = self._memory.unpack_state(state)
        state = tf.expand_dims(state, axis=0)
        action = self.actor.model.predict(state, verbose=0)[0]
        noise = np.zeros([1, self.n_actions])
        action_p = np.zeros(self.n_actions)

        self.epsilon = self.epsilon - 1 / 100000.0
        noise[0][0] = is_training * max(self.epsilon, 0) * self.OU.function(action[0], 0.0, 0.60, 0.30)
        noise[0][1] = is_training * max(self.epsilon, 0) * self.OU.function(action[1], 0.5, 1.00, 0.10)
        noise[0][2] = is_training * max(self.epsilon, 0) * self.OU.function(action[2], -0.1, 1.00, 0.05)

        action_p[0] = action[0] + noise[0][0]
        action_p[1] = action[1] + noise[0][1]
        action_p[2] = action[2] + noise[0][2]

        # clip the resulting action with the bounds
        action_p = np.clip(action_p, self.lower_bound, self.upper_bound)

        return action_p

    def learn(self):
        """
        Fill the buffer up to the batch size, then train both networks with
        experience from the replay buffer.
        """
        avg_loss = 0
        if self._memory.is_ready(self.batch_size):
            avg_loss = self.train_helper()

        return avg_loss

    def save_models(self):
        self.actor.model.save(self.save_dir + "/actor")
        self.actor.target_model.save(self.save_dir + "/actor_target")
        self.critic.model.save(self.save_dir + "/critic")
        self.critic.target_model.save(self.save_dir + "/critic_target")

    """
    Train helper methods
    train_helper
    train_critic
    train_actor
    get_q_targets  Q values to train the critic
    get_gradients  policy gradients to train the actor
    """

    def train_helper(self):
        # get experience batch
        states, actions, rewards, terminal, states_n = self._memory.sample(self.batch_size)
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        rewards = tf.cast(rewards, dtype=tf.float32)
        states_n = tf.convert_to_tensor(states_n)
        terminal = np.expand_dims(terminal, 1)
        terminal = tf.convert_to_tensor(terminal)

        # train the critic before the actor
        self.train_critic(states, actions, rewards, terminal, states_n)
        actor_loss = self.train_actor(states)
        # update the target models
        self.critic.update_target()
        self.actor.update_target()
        K.clear_session()
        return actor_loss

    def train_critic(self, states, actions, rewards, terminal, states_n):
        """
        Use updated Q targets to train the critic network
        """
        # TODO cleaner code, ugly passing of actor target model
        self.critic.train(states, actions, rewards, terminal, states_n, self.actor.target_model, self.gamma)

    def train_actor(self, states):
        """
        Train the actor network with the critic evaluation
        """
        # TODO cleaner code, ugly passing of critic model
        return self.actor.train(states, self.critic.model)

    def remember(self, state, state_new, action, reward, terminal):
        """
        replay buffer interface to the outsize
        """
        self._memory.remember(state, state_new, action, reward, terminal)
