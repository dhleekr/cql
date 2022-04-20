from collections import deque, namedtuple
import random
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from .networks import Actor, Critic


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, buffer_size=1e6):
        self.buffer = deque([], maxlen=buffer_size)
    
    def push(self, *args): # s, a, r, s', done
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return [state, action, reward, next_state, done]

    def __len__(self):
        return len(self.buffer)


# args contain : hidden_dim, lr, buffer_size, episodes, target_update, batch_size
class SAC:
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.args = args
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.alpha = self.args.alpha

        # Q1, Q2, Q1_target, Q2_target
        self.critic1 = Critic(self.observation_dim, self.action_dim, args.hidden_dim)
        self.critic2 = Critic(self.observation_dim, self.action_dim, args.hidden_dim)
        self.critic1_target = Critic(self.observation_dim, self.action_dim, args.hidden_dim)
        self.critic2_target = Critic(self.observation_dim, self.action_dim, args.hidden_dim)

        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=args.lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=args.lr)
        self.critic1_target_optimizer = Adam(self.critic1_target.parameters(), lr=args.lr)
        self.critic2_target_optimizer = Adam(self.critic2_target.parameters(), lr=args.lr)

        # Actor (continuous action space only now)
        self.actor = Actor(self.env, self.observation_dim, self.action_dim, args.hidden_dim)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.lr)

        # Alpha
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = Adam([self.log_alpha], lr=args.lr)

        self.logger = {
            'total_timesteps' : [],
            'episode_reward' : [],
            'critic_loss' : [],
            'actor_loss' : [],
            'alpha_loss' : [],
            'q_target' : [],
            'q1' : [],
            'q2' : [],
            'log_prob' : []
        }

        print("Initializing...")

    def train(self):
        print("Training...")
        replay_buffer = ReplayBuffer(self.args.buffer_size)
        total_timesteps = 0
        gradient_steps = 0
        
        for i_episode in range(self.args.num_episodes):
            episode_reward = 0
            episode_steps = 0
            done = False
            state = self.env.reset()

            while not done:
                if self.args.render and i_episode % self.args.render_period == 0:
                    self.env.render()

                if len(replay_buffer) > self.args.batch_size:
                    for _ in range(self.args.updates):
                        batch = replay_buffer.sample(self.args.batch_size)
                        self.update(batch, gradient_steps)
                        gradient_steps += 1

                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)

                total_timesteps += 1
                episode_reward += reward
                episode_steps += 1
                state = next_state

            self.logger['total_timesteps'].append(total_timesteps)
            self.logger['episode_reward'].append(episode_reward)

            if i_episode != 0 and i_episode % self.args.logging_period == 0:
                self.logging()
            
            if total_timesteps > self.args.timesteps:
                break

        print("Finished!!!!")
        
                
    def update(self, batch, gradient_steps):
        state, action, reward, next_state, done = batch
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1)
        reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_state = torch.tensor(next_state, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float).unsqueeze(1)

        # Compute critic loss
        # next states come from replay buffer and next actions come from current policy
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_target = self.critic1_target(next_state, next_action)
            q2_target = self.critic2_target(next_state, next_action)
            q_target = reward + (1 - done) * self.args.gamma * (torch.minimum(q1_target, q2_target) - self.alpha * next_log_prob)

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)

        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        
        critic_loss = q1_loss + q2_loss

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Compute actor loss
        action_from_current, log_prob_from_current = self.actor.sample(state)
        q1_from_current = self.critic1(state, action_from_current)
        q2_from_current = self.critic2(state, action_from_current)
        q_from_current = torch.minimum(q1_from_current, q2_from_current)

        actor_loss = (self.alpha * log_prob_from_current - q_from_current).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Compute alpha loss
        alpha_loss = -(self.log_alpha * (log_prob_from_current.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()
        
        self.logger['critic_loss'].append(critic_loss)
        self.logger['actor_loss'].append(actor_loss)
        self.logger['alpha_loss'].append(alpha_loss)
        self.logger['q_target'].append(torch.sum(q_target)/len(q_target))
        self.logger['q1'].append(torch.sum(q1)/len(q1))
        self.logger['q2'].append(torch.sum(q2)/len(q2))
        self.logger['log_prob'].append(torch.sum(log_prob_from_current)/len(log_prob_from_current))

        # target_update : soft_target -> 1, hard_target -> 1000
        if gradient_steps % self.args.target_update == 0:
            if self.args.hard_target:
                self.args.tau = 1

            for current_param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data = (1 - self.args.tau) * target_param.data.clone().detach() \
                                    + self.args.tau * current_param.data.clone().detach()

            for current_param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data = (1 - self.args.tau) * target_param.data.clone().detach() \
                                    + self.args.tau * current_param.data.clone().detach()

            
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def logging(self):
        total_timesteps = self.logger['total_timesteps'][0]
        episode_reward = self.logger['episode_reward'][0]
        avg_critic_loss = sum(self.logger['critic_loss']) / len(self.logger['critic_loss'])
        avg_actor_loss = sum(self.logger['actor_loss']) / len(self.logger['actor_loss'])
        avg_alpha_loss = sum(self.logger['alpha_loss']) / len(self.logger['alpha_loss'])
        avg_q_target = sum(self.logger['q_target']) / len(self.logger['q_target'])
        avg_q1 = sum(self.logger['q1']) / len(self.logger['q1'])
        avg_q2 = sum(self.logger['q2']) / len(self.logger['q2'])
        avg_log_prob = sum(self.logger['log_prob']) / len(self.logger['log_prob'])
        
        print('#'*50)
        print(f"Total timesteps : {total_timesteps}")
        print(f"Episode reward : {episode_reward}")
        print(f"Critic loss : {avg_critic_loss}")
        print(f"Actor loss : {avg_actor_loss}")
        print(f"Alpha loss : {avg_alpha_loss}")
        print(f"Q target : {avg_q_target}")
        print(f"Q1 : {avg_q1}")
        print(f"Q2 : {avg_q2}")
        print(f"Log prob : {avg_log_prob}")
        print('#'*50)
        print('\n')

        for k in self.logger.keys():
            self.logger[k] = []