import numpy as np
from ppo import PPO
from tools import mpi
from tools import log
from config import settings
from tools.env import init_reacher_env
from typing import Tuple, Dict
from argparse import ArgumentParser, Namespace
from unityagents import UnityEnvironment, BrainParameters, BrainInfo


def random_cc() -> None:
    # Init environment.
    env, brain_name, state_size, action_size, state = init_reacher_env()

    # Take random actions in the environment.
    score: float = 0                                            # initialize the score

    while True:
        action: np.ndarray = np.random.randn(action_size)       # select an action
        action = np.clip(action, -1, 1)                         # all actions between -1 and 1
        env_info: BrainInfo = env.step(action)[brain_name]      # send the action to the environment
        reward: float = env_info.rewards[0]                     # get the reward
        done: bool = env_info.local_done[0]                     # see if episode has finished
        score += reward                                         # update the score
        if done:                                                # exit loop if episode finished
            env.close()
            break

    print("Score: {}".format(score))


def train() -> None:
    """# Init environment.
    env, brain_name, state_size, action_size, state = init_env(settings.env_file, train_mode=True)

    agent = ActorCritic(state_size, action_size, 24)

    print("Actor model")
    print("============================================================")
    print(agent.actor)

    print("Critic model")
    print("============================================================")
    print(agent.critic)

    env.close()"""


def main():
    # Get arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="reach-ppo")
    args: Namespace = parser.parse_args()

    # Run parallel code with MPI
    mpi.mpi_fork(settings.cores)

    # Get logging kwargs
    logger_kwargs: Dict[str, str] = log.setup_logger_kwargs(args.exp_name, settings.seed, settings.out_dir, True)

    ppo: PPO = PPO(
        env_fn=init_reacher_env,
        seed=settings.seed,
        steps_per_epoch=settings.PPO.steps_per_epoch,
        epochs=settings.PPO.epochs,
        gamma=settings.PPO.gamma,
        clip_ratio=settings.PPO.clip_ratio,
        policy_lr=settings.ActorCritic.policy_lr,
        value_lr=settings.ActorCritic.value_lr,
        train_policy_iters=settings.ActorCritic.train_policy_iters,
        train_value_iters=settings.ActorCritic.train_value_iters,
        lam=settings.PPO.lam,
        max_ep_len=settings.PPO.max_ep_len,
        target_kl=settings.PPO.target_kl,
        logger_kwargs=logger_kwargs,
        save_freq=settings.save_freq
    )

    ppo.train()


if __name__ == "__main__":
    main()
