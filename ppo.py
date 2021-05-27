import torch
import numpy as np
from unityagents import UnityEnvironment

from buffer import PPOBuffer
from tools import mpi, nets
from config import settings
from agents import ActorCritic
from tools.log import EpochLogger
from typing import Dict, Callable, Tuple


class PPO:
    def __init__(
            self,
            env_fn: Callable[[int], Tuple[UnityEnvironment, str, int, int, Tuple[float]]],
            seed: int,
            steps_per_epoch: int,
            gamma: float,
            lam: float,
            logger_kwargs: Dict[str, str]
    ):
        """
        Proximal Policy Optimization (by clipping),
        with early stopping based on approximate KL
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.
            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to PPO.
            seed: Seed for random number generators.
            steps_per_epoch: Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.
            epochs (int): Number of epochs of interaction (equivalent to
                number of policy updates) to perform.
            gamma: Discount factor. (Always between 0 and 1.)
            clip_ratio (float): Hyperparameter for clipping in the policy objective.
                Roughly: how far can the new policy go from the old policy while
                still profiting (improving the objective function)? The new policy
                can still go farther than the clip_ratio says, but it doesn't help
                on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
                denoted by :math:`\epsilon`.
            pi_lr (float): Learning rate for policy optimizer.
            vf_lr (float): Learning rate for value function optimizer.
            train_pi_iters (int): Maximum number of gradient descent steps to take
                on policy loss per epoch. (Early stopping may cause optimizer
                to take fewer than this.)
            train_v_iters (int): Number of gradient descent steps to take on
                value function per epoch.
            lam: Lambda for GAE-Lambda. (Always between 0 and 1,
                close to 1.)
            max_ep_len (int): Maximum length of trajectory / episode / rollout.
            target_kl (float): Roughly what KL divergence we think is appropriate
                between new and old policies after an update. This will get used
                for early stopping. (Usually small, 0.01 or 0.05.)
            logger_kwargs: Keyword args for EpochLogger.
            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.
        """

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        mpi.setup_pytorch_for_mpi()

        # Set up logger and save configuration
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(settings.as_dict())

        # Random seed
        seed += 10000 * mpi.proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instantiate environment
        env, brain_name, state_size, action_size, state = env_fn(seed)

        # Create actor-critic module
        ac = ActorCritic(state_size=state_size, action_size=action_size, seed=seed)

        # Sync params across processes
        mpi.sync_params(ac)

        # Count variables
        var_counts = tuple(nets.count_vars(module) for module in [ac.pi, ac.v])
        logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # Set up experience buffer
        local_steps_per_epoch = int(steps_per_epoch / mpi.num_procs())
        buf = PPOBuffer(state_size, action_size, local_steps_per_epoch, gamma, lam)

        print(f"buffer {mpi.proc_id()}")
        print(buf)

        """if mpi.proc_id() in (1, 2):
            for p in ac.parameters():
                print(f"ActorCritic params proc {mpi.proc_id()}")
                print("============================================================")
                print(p.data.shape)
                print(p.data)"""

        env.close()

