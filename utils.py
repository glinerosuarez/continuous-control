from enum import Enum


class Activation(Enum):

    @classmethod
    def get_values(cls):
        return [a.value for a in cls.__members__.values()]

    TANH = "tanh"
    RELU = "relu"
