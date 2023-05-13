import os
import time
import logging
import argparse

import numpy as np
import tensorflow as tf
from gym.spaces import Box

from agents.tf2rl.experiments.utils import save_path, frames_to_gif
from agents.tf2rl.misc.get_replay_buffer import get_replay_buffer
from agents.tf2rl.misc.prepare_output_dir import prepare_output_dir
from agents.tf2rl.misc.initialize_logger import initialize_logger
from agents.tf2rl.envs.normalizer import EmpiricalNormalizer

from torcs_client.reward import LocalReward

if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
        tf.config.experimental.set_memory_growth(cur_device, enable=True)


def unpack_state(state):
    """
    state dict to state array if fixed order
    """
    state_array = np.zeros(410)

    state_array[0:400] = state["midLine"]
    state_array[400] = state["trackPos"]
    state_array[401] = state["speedX"]
    state_array[402] = state["speedY"]
    state_array[403] = state["speedZ"]
    state_array[404] = state["accX"]
    state_array[405] = state["accY"]
    state_array[406] = state["accZ"]
    state_array[407] = state["angle"]
    state_array[408] = state["yaw_rate"]
    state_array[409] = state["rpm"]

    track_sensor = state["track"]

    game_info = dict()
    game_info["distRaced"] = state["distRaced"]
    game_info["trackLen"] = state["trackLen"]
    game_info["totalTime"] = state["totalTime"]

    return state_array, track_sensor, game_info


class Trainer:
    def __init__(self, policy, env, args, test_env=None):
        self.prev_accel = 0
        if isinstance(args, dict):
            _args = args
            args = policy.__class__.get_argument(Trainer.get_argument())
            args = args.parse_args([])
            for k, v in _args.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                else:
                    raise ValueError(f"{k} is invalid parameter.")

        self._set_from_args(args)
        self._policy = policy
        self._env = env
        self._test_env = self._env if test_env is None else test_env
        if self._normalize_obs:
            assert isinstance(env.observation_space, Box)
            self._obs_normalizer = EmpiricalNormalizer(
                shape=env.observation_space.shape)

        # prepare log directory
        self._output_dir = prepare_output_dir(
            args=args, user_specified_dir=self._logdir,
            suffix="{}_{}".format(self._policy.policy_name, args.dir_suffix))
        self.logger = initialize_logger(
            logging_level=logging.getLevelName(args.logging_level),
            output_dir=self._output_dir)

        if args.evaluate:
            assert args.model_dir is not None
        self._set_check_point(args.model_dir)

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self._output_dir)
        self.writer.set_as_default()

        self._best_train_duration = 1000000
        self._best_test_duration = 1000000

    def _set_check_point(self, model_dir):
        # Save and restore model
        self._checkpoint = tf.train.Checkpoint(policy=self._policy)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=self._output_dir, max_to_keep=25)

        if model_dir is not None:
            assert os.path.isdir(model_dir)
            self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self._checkpoint.restore(self._latest_path_ckpt)
            self.logger.info("Restored {}".format(self._latest_path_ckpt))

    def simple_controller(self, state, track_sensor):
        action = np.zeros(2)

        speedX = abs(state[401] * 300)
        # steer to corner
        steer = state[407] * 19
        # # steer to center
        steer -= state[400] * .4

        if track_sensor[9] < 0.3 and speedX > 55:
            # front is getting close (80 mt)
            accel = -0.8
        else:
            accel = self.prev_accel

        if speedX < 120 - (steer * 50):
            accel += .03
        else:
            accel -= .03

        if accel > 0.5:
            accel = 0.5

        if speedX < 10:
            accel += 1 / (speedX + .1)

        if accel >= 0:
            self.prev_accel = accel

        action[0] = steer
        action[1] = accel

        return action

    def __call__(self, track_list):

        returns = []
        steps = []

        for track in track_list:
            if self._evaluate:
                self.evaluate_policy_continuously()

            total_steps = 0
            tf.summary.experimental.set_step(total_steps)
            episode_steps = 0
            episode_return = 0
            n_episode = 0

            replay_buffer = get_replay_buffer(
                self._policy, self._env, self._use_prioritized_rb,
                self._use_nstep_rb, self._n_step)
            self._env.set_track(track)

            obs = self._env.reset()
            obs, track_sensor, game_info = unpack_state(obs)
            episode_start_time = time.perf_counter()
            while total_steps < self._max_steps:
                if total_steps < self._policy.n_warmup:
                    action = self._env.action_space.sample()
                else:
                    action = self._policy.get_action(obs)

                # if n_episode <= 1 and episode_steps < 3000 or n_episode % 15 == 0:
                #     action = self.simple_controller(obs, track_sensor)
                next_obs, reward, done = self._env.step(action)
                next_obs, track_sensor, game_info = unpack_state(next_obs)

                episode_steps += 1
                episode_return += reward
                total_steps += 1
                tf.summary.experimental.set_step(total_steps)

                done_flag = done

                replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag)
                obs = next_obs

                if done:
                    duration = time.perf_counter() - episode_start_time
                    fps = episode_steps / duration

                    if game_info["distRaced"] >= (game_info["trackLen"] - 2) and \
                            duration < self._best_train_duration:
                        self._best_train_duration = game_info["totalTime"]
                        self.logger.info("Saving checkpoint")
                        self.logger.info("Best duration: {}".format(self._best_train_duration))
                        self.checkpoint_manager.save(total_steps)

                    self.logger.info(
                        "Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} TIME(s): {4:6.2f} "
                        "FPS: {5:5.2f}".format(
                            n_episode + 1, int(total_steps), episode_steps, episode_return, duration, fps))
                    tf.summary.scalar(name="Common/training_return", data=episode_return)
                    tf.summary.scalar(name="Common/training_episode_length", data=episode_steps)

                    replay_buffer.on_episode_end()
                    obs = self._env.reset()
                    obs, track_sensor, game_info = unpack_state(obs)

                    if n_episode > 500:
                        returns.append(episode_return)
                        steps.append(episode_steps)

                    n_episode += 1

                    episode_steps = 0
                    episode_return = 0

                if total_steps < self._policy.n_warmup:
                    continue

                if total_steps % self._policy.update_interval == 0:
                    samples = replay_buffer.sample(self._policy.batch_size)
                    with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                        self._policy.train(
                            samples["obs"], samples["act"], samples["next_obs"],
                            samples["rew"], np.array(samples["done"], dtype=np.float32),
                            None if not self._use_prioritized_rb else samples["weights"])
                    if self._use_prioritized_rb:
                        td_error = self._policy.compute_td_error(
                            samples["obs"], samples["act"], samples["next_obs"],
                            samples["rew"], np.array(samples["done"], dtype=np.float32))
                        replay_buffer.update_priorities(
                            samples["indexes"], np.abs(td_error) + 1e-6)

                if total_steps % self._test_interval == 0:
                    avg_test_return, avg_test_steps, test_duration, finished = self.evaluate_policy(total_steps)
                    self.logger.info(
                        "Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} duration: {2:6.2f} finished: {3}".format(
                            total_steps, avg_test_return, test_duration, finished))
                    fps = avg_test_steps / test_duration
                    tf.summary.scalar(
                        name="Common/average_test_return", data=avg_test_return)
                    tf.summary.scalar(
                        name="Common/average_test_episode_length", data=avg_test_steps)
                    tf.summary.scalar(name="Common/fps", data=fps)

                    if finished and test_duration < self._best_test_duration:
                        self._best_test_duration = test_duration
                        self.logger.info("Saving checkpoint")
                        self.logger.info("Best duration: {}".format(self._best_test_duration))
                        self.checkpoint_manager.save(total_steps)

                if done:
                    episode_start_time = time.perf_counter()

            if n_episode > 500:
                returns.append(episode_return)
                steps.append(episode_steps)

        tf.summary.flush()

        return returns, steps, []

    def test(self):
        self._env.set_track("Zongxoi-city")
        while True:
            obs = self._env.reset()
            obs, track_sensor, game_info = unpack_state(obs)
            done = False
            start_time = time.perf_counter()
            while not done:
                action = self._policy.get_action(obs, test=1.0)
                next_obs, reward, done = self._env.step(action)
                next_obs, track_sensor, game_info = unpack_state(next_obs)
                obs = next_obs
            end_time = time.perf_counter()
            print("Duration: {}".format(end_time - start_time))

    def evaluate_policy_continuously(self):
        """
        Periodically search the latest checkpoint, and keep evaluating with the latest model until user kills process.
        """
        if self._model_dir is None:
            self.logger.error("Please specify model directory by passing command line argument `--model-dir`")
            exit(-1)

        self.evaluate_policy(total_steps=0)
        while True:
            latest_path_ckpt = tf.train.latest_checkpoint(self._model_dir)
            if self._latest_path_ckpt != latest_path_ckpt:
                self._latest_path_ckpt = latest_path_ckpt
                self._checkpoint.restore(self._latest_path_ckpt)
                self.logger.info("Restored {}".format(self._latest_path_ckpt))
            self.evaluate_policy(total_steps=0)

    def evaluate_policy(self, total_steps):
        tf.summary.experimental.set_step(total_steps)

        avg_test_return = 0.
        avg_test_steps = 0
        finished = False

        for i in range(1):
            episode_return = 0.
            obs = self._test_env.reset()
            obs, _, game_info = unpack_state(obs)
            avg_test_steps += 1
            start_time = time.perf_counter()
            end_time = 0
            for _ in range(1000000):
                action = self._policy.get_action(obs, test=True)
                next_obs, reward, done = self._test_env.step(action)
                next_obs, _, game_info = unpack_state(next_obs)
                avg_test_steps += 1

                episode_return += reward
                obs = next_obs
                if done:
                    end_time = time.perf_counter()
                    if game_info["distRaced"] >= (game_info["trackLen"] - 2):
                        finished = True
                    break
            test_duration = end_time - start_time
            avg_test_return += episode_return

        return avg_test_return, avg_test_steps, test_duration, finished

    def _set_from_args(self, args):
        # experiment settings
        self._max_steps = args.max_steps
        self._episode_max_steps = (args.episode_max_steps
                                   if args.episode_max_steps is not None
                                   else args.max_steps)
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        self._save_model_interval = args.save_model_interval
        self._save_summary_interval = args.save_summary_interval
        self._normalize_obs = args.normalize_obs
        self._logdir = args.logdir
        self._model_dir = args.model_dir
        # replay buffer
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step
        # test settings
        self._evaluate = args.evaluate
        self._test_interval = args.test_interval
        self._show_test_progress = args.show_test_progress
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path
        self._save_test_movie = args.save_test_movie
        self._show_test_images = args.show_test_images

    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # experiment settings
        parser.add_argument('--max-steps', type=int, default=int(1e6),
                            help='Maximum number steps to interact with env.')
        parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                            help='Maximum steps in an episode')
        parser.add_argument('--n-experiments', type=int, default=1,
                            help='Number of experiments')
        parser.add_argument('--show-progress', action='store_true',
                            help='Call `render` in training process')
        parser.add_argument('--save-model-interval', type=int, default=int(1e4),
                            help='Interval to save model')
        parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
                            help='Interval to save summary')
        parser.add_argument('--model-dir', type=str, default=None,
                            help='Directory to restore model')
        parser.add_argument('--dir-suffix', type=str, default='',
                            help='Suffix for directory that contains results')
        parser.add_argument('--normalize-obs', action='store_true',
                            help='Normalize observation')
        parser.add_argument('--logdir', type=str, default='results',
                            help='Output directory')
        # test settings
        parser.add_argument('--evaluate', action='store_true',
                            help='Evaluate trained model')
        parser.add_argument('--test-interval', type=int, default=int(1e4),
                            help='Interval to evaluate trained model')
        parser.add_argument('--show-test-progress', action='store_true',
                            help='Call `render` in evaluation process')
        parser.add_argument('--test-episodes', type=int, default=5,
                            help='Number of episodes to evaluate at once')
        parser.add_argument('--save-test-path', action='store_true',
                            help='Save trajectories of evaluation')
        parser.add_argument('--show-test-images', action='store_true',
                            help='Show input images to neural networks when an episode finishes')
        parser.add_argument('--save-test-movie', action='store_true',
                            help='Save rendering results')
        # replay buffer
        parser.add_argument('--use-prioritized-rb', action='store_true',
                            help='Flag to use prioritized experience replay')
        parser.add_argument('--use-nstep-rb', action='store_true',
                            help='Flag to use nstep experience replay')
        parser.add_argument('--n-step', type=int, default=4,
                            help='Number of steps to look over')
        # others
        parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],
                            default='INFO', help='Logging level')
        return parser
