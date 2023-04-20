import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

from torcs_client.torcs_comp import TorcsEnv


def main(verbose=False, hyperparams=None, sensors=None, image_name="zjlqwq/gym_torcs:v1.2", driver=None,
         privileged=False, training=None, algo_name=None, algo_path=None, stack_depth=1, img_width=640, img_height=480):
    # PPO epochs
    n_epochs = 5
    if "epochs" in training.keys(): n_epochs = training["epochs"]

    # track and car selection
    track_list = [None]
    car = None
    if "track" in training.keys(): track_list = training["track"]
    if "car" in training.keys(): car = training["car"]

    # UDP port selection(Don't change this)
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
    # Env for testing
    test_env = env

    args = {"test_episodes": 1, "test_interval": hyperparams["test_interval"],
            "save_summary_interval": hyperparams["test_interval"], "save_model_interval": hyperparams["test_interval"],
            "max_steps": training["max_steps"], "episode_max_steps": int(1e7), "dir_suffix": ""}

    if "model_dir" in hyperparams.keys():
        args["model_dir"] = hyperparams["model_dir"]

    if training["algo"] == "PPO":
        from agents.tf2rl.algos.ppo import PPO
        from agents.tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
        from agents.tf2rl.experiments.utils import load_expert_traj

        agent = PPO(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            is_discrete=False,
            max_action=env.action_space.high[0],
            batch_size=hyperparams["batch_size"],
            actor_units=(hyperparams["fcl1_size"], hyperparams["fcl2_size"]),
            critic_units=(hyperparams["fcl1_size"], hyperparams["fcl2_size"]),
            n_epoch=n_epochs,
            lr_actor=hyperparams["actor_lr"],
            lr_critic=hyperparams["critic_lr"],
            hidden_activation_actor="tanh",
            hidden_activation_critic="tanh",
            discount=hyperparams["gamma"],
            lam=hyperparams["lam"],
            clip_ratio=hyperparams["clip_ratio"],
            vfunc_coef=hyperparams["c_1"],
            entropy_coef=hyperparams["c_2"],
            horizon=hyperparams["horizon"]
        )

        expert_trajs = None

        if "dataset_dir" in hyperparams.keys():
            expert_trajs = load_expert_traj(hyperparams["dataset_dir"])

        trainer = OnPolicyTrainer(agent, env, args, test_env=test_env, expert_trajs=expert_trajs)

    elif training["algo"] == "DDPG":
        from agents.tf2rl.algos.ddpg import DDPG
        from agents.tf2rl.experiments.trainer import Trainer

        agent = DDPG(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            memory_capacity=hyperparams["buf_size"],
            max_action=env.action_space.high[0],
            batch_size=hyperparams["batch_size"],
            actor_units=(hyperparams["fcl1_size"], hyperparams["fcl2_size"]),
            critic_units=(hyperparams["fcl1_size"], hyperparams["fcl2_size"]),
            lr_actor=hyperparams["actor_lr"],
            tau=hyperparams["tau"],
            lr_critic=hyperparams["critic_lr"],
            n_warmup=hyperparams["n_warmup"],
            update_interval=hyperparams["update_interval"]
        )

        trainer = Trainer(agent, env, args, test_env=test_env)

    elif training["algo"] == "SAC":
        from agents.tf2rl.algos.sac import SAC
        from agents.tf2rl.experiments.trainer import Trainer

        agent = SAC(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            memory_capacity=hyperparams["buf_size"],
            max_action=env.action_space.high[0],
            batch_size=hyperparams["batch_size"],
            actor_units=(hyperparams["fcl1_size"], hyperparams["fcl2_size"]),
            critic_units=(hyperparams["fcl1_size"], hyperparams["fcl2_size"]),
            tau=hyperparams["tau"],
            n_warmup=hyperparams["n_warmup"],
            update_interval=hyperparams["update_interval"]
        )
        trainer = Trainer(agent, env, args, test_env=test_env)

    # returns, steps, entropies = trainer(track_list)

    trainer.test()
    # plotting
    matplotlib.use("Agg")

    fig, ax = plt.subplots()

    ax.set(xlabel="Episode", ylabel="Return", title="Return per episode")
    ax.plot([x for x in range(len(returns))], returns)

    fig.savefig("plots/returns_{}_{}.png".format(training["algo"], datetime.today().strftime("%h%m%d%m%Y")))

    fig, ax = plt.subplots()
    ax.set(xlabel="Episode", ylabel="Steps", title="Total steps per episode")
    ax.plot([x for x in range(len(steps))], steps)

    with open("plots/returns_{}.txt".format(training["algo"]), "w") as f:
        for item in returns:
            f.write("{}\n".format(item))

    with open("plots/steps_{}.txt".format(training["algo"]), "w") as f:
        for item in steps:
            f.write("{}\n".format(item))

    fig.savefig("plots/steps_{}_{}.png".format(training["algo"], datetime.today().strftime("%h%m%d%m%Y")))

    # plt.show()

    if len(entropies) > 0:
        fig, ax = plt.subplots()
        ax.plot([x for x in range(len(entropies))], entropies)

        ax.set(xlabel="Training step", ylabel="Entropy", title="Entropy loss during training")

        with open("plots/entropies_{}.txt".format(training["algo"]), "w") as f:
            for item in entropies:
                f.write("{}\n".format(item))

        fig.savefig("plots/entropies_{}_{}.png".format(training["algo"], datetime.today().strftime("%h%m%d%m%Y")))

        # plt.show()

    input("All done")
