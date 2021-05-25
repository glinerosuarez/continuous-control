from enum import Enum
from collections import OrderedDict
import os.path as osp
import numpy as np
import time
from torch.nn import Module, Linear, Tanh, ReLU, Sequential
from typing import Dict


class Activation(Enum):

    @classmethod
    def get_values(cls):
        return [a.value for a in cls.__members__.values()]

    TANH = "tanh"
    RELU = "relu"


def build_model(state_size: int, n_layers: int, hidden_nodes: int, activation: Activation, o_dim: int) -> Sequential:
    """Build a MLP.
    Params
    ======
        state_size:     Size of the states
        n_layers:       Number of hidden layers
        hidden_nodes:   Number of nodes per hidden layer
        activation:     Type of activations
        o_dim:          output dimension
    """
    layers: OrderedDict[str, Module] = OrderedDict()

    for i in range(n_layers):
        if i == 0:
            layers["Linear1"] = Linear(state_size, hidden_nodes)
            layers["Activation1"] = Tanh() if activation == Activation.TANH else ReLU()
        elif i == (n_layers - 1):
            layers[f"Linear{n_layers}"] = Linear(hidden_nodes, o_dim)
            layers[f"Activation{n_layers}"] = Tanh() if activation == Activation.TANH else ReLU()
        else:
            layers[f"Linear{i + 1}"] = Linear(hidden_nodes, hidden_nodes)
            layers[f"Activation{i + 1}"] = Tanh() if activation == Activation.TANH else ReLU()

    return Sequential(layers)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def setup_logger_kwargs(exp_name: str, seed: int = None, data_dir: str = None, datestamp: bool = False) -> Dict[str, str]:
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs.
    If no seed is given and datestamp is false,
    ::
        output_dir = data_dir/exp_name
    If a seed is given and datestamp is false,
    ::
        output_dir = data_dir/exp_name/exp_name_s[seed]
    If datestamp is true, amend to
    ::
        output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]
    Args:
        exp_name: Name for experiment.
        seed: Seed for random number generators used by experiment.
        data_dir: Path to folder where results should be saved.
        datestamp: Whether to include a date and timestamp in the name of the save directory.
    Returns:
        logger_kwargs, a dict containing output_dir and exp_name.
    """

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])

    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), exp_name=exp_name)
    return logger_kwargs
