import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def weights_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        module.bias.data.fill_(0.0)


class Actor_continuous(nn.Module):
    def __init__(self, env, observation_dim, action_dim, hidden_dim):
        super(Actor_continuous, self).__init__()
        self.mlp = MLP(observation_dim, hidden_dim[-1], hidden_dim).to(device)
        self.mean_layer = nn.Linear(hidden_dim[-1], action_dim).to(device)
        self.log_std_layer = nn.Linear(hidden_dim[-1], action_dim).to(device)

        action_space = env.action_space
        self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(device)
        self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(device)

        self.apply(weights_init)

    def forward(self, obs):
        out = F.relu(self.mlp(obs))
        mean = self.mean_layer(out)
        log_std = self.log_std_layer(out)
        # std is going to near 0 -> log_prob -> NaN
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t).to(device)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        return action.to(device), log_prob.to(device)

    def log_prob(self, obs, action):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob


class Actor_discrete(nn.Module):
    def __init__(self, env, observation_dim, action_dim, hidden_dim):
        super(Actor_discrete, self).__init__()
        self.mlp = MLP(observation_dim, hidden_dim[-1], hidden_dim).to(device)
        self.mean_layer = nn.Linear(hidden_dim[-1], action_dim).to(device)
        self.noise = torch.Tensor(action_dim).to(device)

        self.apply(weights_init)

    def forward(self, obs):
        out = F.relu(self.mlp(obs))
        out = self.mean_layer(out)
        mean = torch.tanh(out)
        return mean

    def sample(self, obs):
        mean = self.forward(obs)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action.to(device), torch.tensor(0.).to(device)


class Critic(nn.Module): # we need 4 critics (Q1, Q2, Q1_target, Q2_target)
    def __init__(self, observation_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.mlp = MLP(observation_dim + action_dim, 1, hidden_dim).to(device)

        self.apply(weights_init)
    
    def forward(self, obs, action):
        if len(action.shape) == 1:
            action = action.view(-1, 1)
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