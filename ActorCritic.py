
import gym.spaces
from gym.spaces import Box, Discrete

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class Critic(nn.Module): 
    def __init__(self): super().__init__()

class ActorCritic(nn.Module):
    def __init__(self, observation_space: gym.spaces.Space = None, action_space: gym.spaces.Space = None, **ac_kwargs):
        super().__init__()

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

def mlp(sizes, activation, output_activation=nn.Sigmoid):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class MLPCategoricalActor(Actor):    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, observation):
        probs = self.logits_net(observation)
        return Categorical(probs=probs)

    def _log_prob_from_distribution(self, distribution, action):
        return distribution.log_prob(action)

class MLPGaussianActor(Actor):    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, observation):
        probs = self.logits_net(observation)
        return Bernoulli(probs=probs)

    def _log_prob_from_distribution(self, distribution, action):
        probas = distribution.log_prob(action)
        return torch.sum(probas, dim=1)

class MLPCritic(Critic):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(ActorCritic):
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]   

        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)