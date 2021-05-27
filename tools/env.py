import random
from typing import Tuple
from config import settings
from unityagents import UnityEnvironment, BrainParameters


def init_reacher_env(seed: int) -> Tuple[UnityEnvironment, str, int, int, Tuple[float]]:
    return init_env(settings.env_file, train_mode=settings.train_mode, seed=seed)


def init_env(
        env_file: str,
        train_mode: bool,
        seed: int,
) -> Tuple[UnityEnvironment, str, int, int, Tuple[float]]:
    """initialize UnityEnvironment"""

    env: UnityEnvironment = UnityEnvironment(file_name=env_file, worker_id=random.randint(1, 100), seed=seed)

    # Environments contain brains which are responsible for deciding the actions of their associated agents.
    # Here we check for the first brain available, and set it as the default brain we will be controlling from Python.
    brain_name: str = env.brain_names[0]
    brain: BrainParameters = env.brains[brain_name]

    # Accumulate experience and train the agent.
    action_size: int = brain.vector_action_space_size                                       # number of actions
    state: Tuple[float] = env.reset(train_mode)[brain_name].vector_observations[0]          # initial state
    state_size: int = len(state)                                                            # get the current stat

    return env, brain_name, state_size, action_size, state
