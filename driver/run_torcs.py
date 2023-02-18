import numpy as np
import collections
import time
import os

from torcs_client.torcs_comp import TorcsEnv
from torcs_client.utils import SimpleLogger as log, resize_frame, agent_from_module

MAX_STEPS = 100000


def main(verbose=False, hyperparams=None, sensors=None, image_name="zjlqwq/gym_torcs:v1.0", driver=None,
         privileged=False, training=None, algo_name=None, algo_path=None, stack_depth=1, img_width=640, img_height=480):
    max_steps = 1000
    episodes = 1000
    track_list = [None]
    car = None
    is_training = 1

    if "max_steps" in training.keys(): max_steps = training["max_steps"]
    if "episodes" in training.keys(): episodes = training["episodes"]
    if "track" in training.keys(): track_list = training["track"]
    if "car" in training.keys(): car = training["car"]

    if driver is not None:
        sid = driver["sid"]
        ports = driver["ports"]
        driver_id = driver["index"]
        driver_module = driver["module"]
    else:
        sid = "SCR"
        ports = [3001]
        driver_id = "0"
        driver_module = "scr_server"

    # Instantiate the environment
    env = TorcsEnv(throttle=training["throttle"], gear_change=training["gear_change"], car=car,
                   verbose=verbose, state_filter=sensors, target_speed=training["target_speed"], sid=sid,
                   ports=ports, driver_id=driver_id, driver_module=driver_module, image_name=image_name,
                   privileged=privileged, img_width=img_width, img_height=img_height)

    action_dims = [env.action_space.shape[0]]
    state_dims = [env.observation_space.shape[0]]  # sensors input
    action_boundaries = [env.action_space.low[0], env.action_space.high[0]]

    agent_class = agent_from_module(algo_name, algo_path)

    agent = agent_class(state_dims=state_dims, action_dims=action_dims,
                        action_boundaries=action_boundaries, hyperparams=hyperparams)

    # 保存每个回合的得分
    scores = []

    for track in track_list:
        log.info("Starting {} episodes on track {}".format(episodes, track))
        env.set_track(track)
        for i in range(episodes):
            state = env.reset()
            # 这个回合是否结束的标志
            terminal = False
            # 这个回合的得分
            score = 0
            # 该回合走的步数
            curr_step = 0

            log.info("Episode {}/{} started".format(i + 1, episodes))

            while not terminal and (curr_step < max_steps):
                # predict new action
                action = agent.get_action(state, i, track, is_training)
                # perform the transition according to the chosen action
                state_new, reward, terminal = env.step(action)
                # store the transaction in the memory
                agent.remember(state, state_new, action, reward, terminal)

                # train the agent
                time_start = time.time()
                loss = agent.learn()
                time_end = time.time()
                log.info("time cost: {} ms , loss {}".format(1000 * (time_end - time_start), loss))

                curr_step += 1
                score += reward
                state = state_new

            # 每30回合保存一次模型
            if (i % 30) and (i > 0) == 0:
                agent.save_models()
                log.info("Saving models...")
            scores.append(score)
            log.info("Episode {}/{} finished. Score {:.2f}. Running average {:.2f}".format(i + 1, episodes, score,
                                                                                           np.mean(scores)))

    log.info("All done. Closing...")
    env.terminate()
    input("...")
