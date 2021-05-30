import random
from typing import Tuple
from config import settings
from unityagents import UnityEnvironment, BrainParameters

from tools import mpi


def init_reacher_env(seed: int) -> Tuple[UnityEnvironment, str, int, int, Tuple[float]]:
    """
    Init Reacher environment for training.
    :param seed: random seed.
    :return: Environment initial data.
    """
    return init_env(settings.env_file, train_mode=True, worker_id=mpi.proc_id(), seed=seed)


def init_env(
        env_file: str,
        train_mode: bool,
        worker_id: int,
        seed: int,
) -> Tuple[UnityEnvironment, str, int, int, Tuple[float]]:
    """initialize UnityEnvironment"""

    env: UnityEnvironment = UnityEnvironment(file_name=env_file, worker_id=worker_id, seed=seed)

    # Environments contain brains which are responsible for deciding the actions of their associated agents.
    # Here we check for the first brain available, and set it as the default brain we will be controlling from Python.
    brain_name: str = env.brain_names[0]
    brain: BrainParameters = env.brains[brain_name]

    # Accumulate experience and train the agent.
    action_size: int = brain.vector_action_space_size                                       # number of actions
    state: Tuple[float] = env.reset(train_mode)[brain_name].vector_observations[0]          # initial state
    state_size: int = len(state)                                                            # get the current stat

    return env, brain_name, state_size, action_size, state
