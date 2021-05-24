import torch
from utils import Activation
from config import settings
from collections import OrderedDict
from torch.nn import Sequential, Linear, Module, Tanh, ReLU


class ActorCritic(Module):

    def __init__(self, state_size: int, action_size: int, seed: int) -> None:
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        def build_model(n_layers: int, hidden_nodes: int, activation: Activation, o_dim: int) -> Sequential:
            """Build a MLP.
            Params
            ======
                n_layers: Number of hidden layers
                hidden_nodes: Number of nodes per hidden layer
                activation: Type of activations
                o_dim: output dimension
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
                    layers[f"Linear{i + 1}"] = Linear(hidden_nodes, settings.ActorCritic.hidden_nodes)
                    layers[f"Activation{i + 1}"] = Tanh() if activation == Activation.TANH else ReLU()

            return Sequential(layers)

        super(ActorCritic, self).__init__()

        self.seed: int = seed
        torch.manual_seed(seed)

        self.actor = build_model(
            n_layers=settings.ActorCritic.layers,
            hidden_nodes=settings.ActorCritic.hidden_nodes,
            activation=Activation.TANH if settings.ActorCritic.activation == Activation.TANH.value else Activation.RELU,
            o_dim=action_size
        )

        self.critic = build_model(
            n_layers=settings.ActorCritic.layers,
            hidden_nodes=settings.ActorCritic.hidden_nodes,
            activation=Activation.TANH if settings.ActorCritic.activation == Activation.TANH.value else Activation.RELU,
            o_dim=1
        )

    def forward(self, x):
        raise NotImplementedError
