import torch
from torch import Tensor
from config import settings
from torch.distributions.normal import Normal
from tools.nets import Activation, build_model
from torch.nn import Parameter, Sequential, Linear, Module, Tanh, ReLU


class Actor(Module):
    def __init__(self, state_size: int, action_size: int, seed: int) -> None:
        super(Actor, self).__init__()

        self.seed: int = seed
        torch.manual_seed(seed)

        self.log_std: Parameter = Parameter(-0.5 * torch.ones(action_size, dtype=torch.float32))
        self.mu_net: Sequential = build_model(
            state_size=state_size,
            n_layers=settings.ActorCritic.layers,
            hidden_nodes=settings.ActorCritic.hidden_nodes,
            activation=Activation.TANH if settings.ActorCritic.activation == Activation.TANH.value else Activation.RELU,
            o_dim=action_size
        )

    def distribution(self, obs) -> Normal:
        mu: Tensor = self.mu_net(obs)
        std: Tensor = torch.exp(self.log_std)
        return Normal(mu, std)

    def log_prob_from_dist(self, pi, act):
        # TODO: why sum is required here?
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self.distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob_from_dist(pi, act)
        return pi, logp_a


class Critic(Module):
    def __init__(self, state_size: int, action_size: int, seed: int):
        super(Critic, self).__init__()

        self.v_net: Sequential = build_model(
            state_size=state_size,
            n_layers=settings.ActorCritic.layers,
            hidden_nodes=settings.ActorCritic.hidden_nodes,
            activation=Activation.TANH if settings.ActorCritic.activation == Activation.TANH.value else Activation.RELU,
            o_dim=1
        )

    def forward(self, obs: Tensor) -> Tensor:
        return torch.squeeze(self.v_net(obs), -1)


class ActorCritic(Module):

    def __init__(self, state_size: int, action_size: int, seed: int) -> None:
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        super(ActorCritic, self).__init__()

        self.seed: int = seed
        torch.manual_seed(seed)

        self.pi = Actor(state_size, action_size, seed)
        self.v = Critic(state_size, action_size, seed)

    def step(self, obs: Tensor):
        with torch.no_grad():
            pi = self.pi.distribution(obs)
            a = pi.sample()
            logp_a = self.pi.log_prob_from_dist(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]



