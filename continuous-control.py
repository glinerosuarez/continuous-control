import random
import numpy as np
from tools import mpi
from tools import log
from config import settings
from typing import Tuple, Dict
from argparse import ArgumentParser, Namespace
from unityagents import UnityEnvironment, BrainParameters, BrainInfo

from ppo import PPO


def init_env(
        env_file: str,
        train_mode: bool = True,
        seed: int = random.randint(0, 100)
) -> Tuple[UnityEnvironment, str, int, int, Tuple[float]]:
    """initialize Banana UnityEnvironment"""

    env: UnityEnvironment = UnityEnvironment(file_name=env_file, worker_id=1, seed=seed)

    # Environments contain brains which are responsible for deciding the actions of their associated agents.
    # Here we check for the first brain available, and set it as the default brain we will be controlling from Python.
    brain_name: str = env.brain_names[0]
    brain: BrainParameters = env.brains[brain_name]

    # Accumulate experience and train the agent.
    action_size: int = brain.vector_action_space_size                                       # number of actions
    state: Tuple[float] = env.reset(train_mode)[brain_name].vector_observations[0]          # initial state
    state_size: int = len(state)                                                            # get the current stat

    return env, brain_name, state_size, action_size, state


def random_cc() -> None:
    # Init environment.
    env, brain_name, state_size, action_size, state = init_env(settings.env_file, train_mode=False)

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

    ppo: PPO = PPO(logger_kwargs=logger_kwargs)


if __name__ == "__main__":
    main()
