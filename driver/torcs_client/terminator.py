import numpy as np

# termination over speed starts after speed_term_start steps
speed_term_start = 50
# km/h, episode terminates if car is running slower than this limit
boring_speed = 1
max_damage = 2000
cul_damage = 0
# tolerance steps for out of track
out_max = 100


def custom_terminal(obs, obs_pre, curr_step):
    terminal = False

    angle = np.cos(obs["angle"])
    speed = obs["speedX"]
    damage = obs["damage"]
    damage_pre = obs_pre["damage"]
    track_pos = obs["trackPos"]

    if np.abs(track_pos) > 1:
        if custom_terminal.first_out == -1:
            custom_terminal.first_out = curr_step
        # terminated if the car is out of track for too long
        if curr_step > custom_terminal.first_out + out_max:
            terminal = True
    else:
        # reset if car is back on track - works for initialization and reset
        custom_terminal.first_out = -1

    # if abs(damage - damage_pre) > 0 and np.abs(track_pos) >= 1:
    #     global cul_damage
    #     cul_damage += abs(damage - damage_pre)
    #     if cul_damage > max_damage:
    #         terminal = True
    #         cul_damage = 0
    if damage > max_damage:
        terminal = True

    if speed_term_start < curr_step:
        # Episode terminates if the agent is too slow
        if speed < boring_speed:
            terminal = True

    if angle < 0:
        # Episode is terminated if the agent runs backward
        terminal = True

    if obs["distRaced"] >= (obs["trackLen"] - 2):
        # completed one lap
        terminal = True
    return terminal
