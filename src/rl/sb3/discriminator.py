# Reference: github.com/Egiob/DiversityIsAllYouNeed-SB3/

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from stable_baselines3.common.utils import obs_as_tensor


class Discriminator(nn.Module):
    """
    MLP-based network that estimates log p(z|s) by having fixed-variance Gaussian
    """
    def __init__(
        self,
        in_sz,
        out_sz,
        hidden_sz=128,
        hidden_depth=2,
        std=0.2,
        device="auto",
        **kwargs
    ):
        super(Discriminator, self).__init__()
        self.device = device

        layers = []
        for i in range(hidden_depth + 1):
            layer_in = in_sz if i == 0 else hidden_sz
            layer_out = out_sz if i == hidden_depth else hidden_sz
            layers.append(nn.Linear(layer_in, layer_out))
            if i != hidden_depth:
                layers.append(nn.ReLU())
        self.net_mu = nn.Sequential(*layers).to(self.device)

        self.out_sz = out_sz
        self.std = std
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, s, a=None, z=None):
        if not isinstance(s, torch.Tensor):
            s = torch.Tensor(s).to(self.device)
        if z is not None and not isinstance(z, torch.Tensor):
            z = torch.Tensor(z).to(self.device)

        mu = self.net_mu(s.to(torch.float32))
        std = torch.tensor(self.std,).to(self.device)

        # Define a Gaussian distribution with fixed variance
        gaussian_distrib = Normal(mu, std)
        
        # If z is not provided, sample from the Gaussian distribution
        if z is None:
            z = gaussian_distrib.rsample()
        
        # Calculate log p(z|s)
        log_prob_z_given_s = gaussian_distrib.log_prob(z)
        return log_prob_z_given_s, mu
    
    def loss(self, obs):
        obs = obs_as_tensor(obs, self.device)
        log_prob_z_given_s, _ = self(obs["proprioception"], z=obs["user_params"])
        return log_prob_z_given_s
    
    def calculate_z(self, obs):
        obs = obs_as_tensor(obs, self.device)
        _, mu = self(obs["proprioception"])
        return(mu.detach().cpu().numpy())
