from enum import Enum
from collections import OrderedDict
import numpy as np
from torch.nn import Module, Linear, Tanh, ReLU, Sequential





def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)
