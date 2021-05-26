import torch
import numpy as np
from unityagents import UnityEnvironment

from tools import mpi
from config import settings
from agents import ActorCritic
from tools.log import EpochLogger
from typing import Dict, Callable, Tuple


class PPO:
    def __init__(
            self,
            env_fn: Callable[[], Tuple[UnityEnvironment, str, int, int, Tuple[float]]],
            seed: int,
            logger_kwargs: Dict[str, str]
    ):
        """
        Proximal Policy Optimization (by clipping),
        with early stopping based on approximate KL
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.
            actor_critic: The constructor method for a PyTorch Module with a
                ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
                module. The ``step`` method should accept a batch of observations
                and return:
                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Numpy array of actions for each
                                               | observation.
                ``v``        (batch,)          | Numpy array of value estimates
                                               | for the provided observations.
                ``logp_a``   (batch,)          | Numpy array of log probs for the
                                               | actions in ``a``.
                ===========  ================  ======================================
                The ``act`` method behaves the same as ``step`` but only returns ``a``.
                The ``pi`` module's forward call should accept a batch of
                observations and optionally a batch of actions, and return:
                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``pi``       N/A               | Torch Distribution object, containing
                                               | a batch of distributions describing
                                               | the policy for the provided observations.
                ``logp_a``   (batch,)          | Optional (only returned if batch of
                                               | actions is given). Tensor containing
                                               | the log probability, according to
                                               | the policy, of the provided actions.
                                               | If actions not given, will contain
                                               | ``None``.
                ===========  ================  ======================================
                The ``v`` module's forward call should accept a batch of observations
                and return:
                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``v``        (batch,)          | Tensor containing the value estimates
                                               | for the provided observations. (Critical:
                                               | make sure to flatten this!)
                ===========  ================  ======================================
            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to PPO.
            seed: Seed for random number generators.
            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.
            epochs (int): Number of epochs of interaction (equivalent to
                number of policy updates) to perform.
            gamma (float): Discount factor. (Always between 0 and 1.)
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
            lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
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
        env, brain_name, state_size, action_size, state = env_fn()

        # Create actor-critic module
        ac = ActorCritic(state_size=state_size, action_size=action_size, seed=seed)

        print("Actor model")
        print("============================================================")
        print(ac.pi)
        print("Critic model")
        print("============================================================")
        print(ac.v)

        env.close()

