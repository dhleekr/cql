import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def weights_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        module.bias.data.fill_(0.01)


class Actor(nn.Module):
    def __init__(self, env, observation_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.mlp = MLP(observation_dim, hidden_dim[-1], hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dim[-1], action_dim)

        action_space = env.action_space
        self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        self.apply(weights_init)

    def forward(self, obs):
        out = F.relu(self.mlp(obs))
        mean = self.mean_layer(out)
        log_std = self.log_std_layer(out)
        # std is going to near 0 -> log_prob -> NaN
        log_std = torch.clamp(log_std, min=-20, max=5)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # # Reparameterization trick (OpenAi spinning up ver.) 
        # normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        # z = normal.sample()
        # action = torch.tanh(mean + std * z)
        # # tanh -> (-1, 1) -> log 0.000000000001 can be possible -> + 1e-7
        # log_prob = Normal(mean, std).log_prob(action) - torch.log(1 - action.pow(2) + 1e-7)

        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module): # we need 4 critics (Q1, Q2, Q1_target, Q2_target)
    def __init__(self, observation_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.mlp = MLP(observation_dim + action_dim, 1, hidden_dim)

        self.apply(weights_init)
    
    def forward(self, obs, action):
        out = torch.cat([obs, action], dim=1)
        return self.mlp(out)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[], activation_fn=nn.ReLU()):
        super(MLP, self).__init__()
        self.output_dim = output_dim

        modules = []
        prev_dim = input_dim
        for dim in hidden_dim:
            modules.append(nn.Linear(prev_dim, dim))
            modules.append(activation_fn)
            prev_dim = dim
        modules.append(nn.Linear(prev_dim, output_dim))
        self.fc_layers = nn.Sequential(*modules)
    
    def forward(self, obs):
        return self.fc_layers(obs)


# maybe this is needed for image-based task later
class CNN(nn.Module):
    pass