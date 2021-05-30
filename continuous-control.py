import torch
import random
import numpy as np
from ppo import PPO
from tools import mpi
from tools import log
from typing import Dict
from pathlib import Path
from config import settings
from agents import ActorCritic
from tools.env import init_env
from unityagents import BrainInfo
from tools.env import init_reacher_env
from argparse import ArgumentParser, Namespace


def random_cc() -> None:
    # Init environment.
    env, brain_name, state_size, action_size, state = init_env(
        settings.env_file, False, random.randint(0, 1000), random.randint(0, 1000)
    )

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


def train(exp_name: str) -> None:
    """
    Implement PPO algorithm to train an ActorCritic agent that solves the Reacher environment.
    :param exp_name: Name of the experiment.
    """

    # Run parallel code with MPI
    mpi.mpi_fork(settings.cores)

    # Get logging kwargs
    logger_kwargs: Dict[str, str] = log.setup_logger_kwargs(exp_name, settings.seed, settings.out_dir, True)

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


def smart_cc(path: str) -> None:
    """
    Take a trained agent to run an episode of the Reacher environment.
    :param path: Path to the dir that stores a trained agent.
    """
    model_path = Path()/path/'pyt_save'/'model.pt'  # Path to saved model file
    state_dicts = torch.load(model_path)            # Load Python dict with state_dicts for ActorCritic agent
    policy_state_dict = state_dicts['policy_state_dict']
    value_state_dict = state_dicts['value_state_dict']

    # Init environment
    seed = random.randint(0, 1000)
    env, brain_name, state_size, action_size, state = init_env(settings.env_file, False, random.randint(0, 1000), seed)

    # Init agent
    agent = ActorCritic(state_size, action_size, seed)

    # Update state dicts
    agent.pi.load_state_dict(policy_state_dict)
    agent.v.load_state_dict(value_state_dict)

    # Run environment
    agent.pi.eval()
    agent.v.eval()
    score = 0.0
    while True:
        action = agent.act(torch.as_tensor(state, dtype=torch.float32))
        env_info: BrainInfo = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        state = next_state
        score += reward
        if done:
            break

    env.close()

    print("Score: {}".format(score))


def main():
    # Get arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-n", "--exp_name", type=str, default="reach-ppo")
    parser.add_argument(
        "-t",
        "--train",
        help="Use PPO algorithm to train an ActorCritic agent that solves the Reacher environment",
        action="store_true"
    )
    parser.add_argument(
        "-c",
        "--continuous_control",
        help="Receives a path to the folder that contains a trained agent to run an epoch in the Reacher environment",
        type=str
    )
    parser.add_argument(
        "-r",
        "--random",
        help="Use a agent that chooses actions at random to run an epoch in the Reacher environment",
        action="store_true"
    )
    args: Namespace = parser.parse_args()

    if args.train:
        train(args.exp_name)
    elif args.random:
        random_cc()
    else:
        smart_cc(args.continuous_control)


if __name__ == "__main__":
    main()
