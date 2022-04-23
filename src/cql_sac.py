from collections import deque, namedtuple
import random
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from .networks import Actor_continuous, Actor_discrete,Critic


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
class CQL_sac:
    def __init__(self, env, dataset, args):
        super().__init__()
        self.env = env
        self.dataset = dataset
        self.args = args
        self.alpha = self.args.alpha
        self.cql_alpha = self.args.cql_alpha
        self.num_samples = 10
        observation_dim = env.observation_space.shape[0]

        # Actor
        if self.args.continuous_space:
            action_dim = env.action_space.shape[0]
            self.actor = Actor_continuous(env, observation_dim, action_dim, args.hidden_dim)
        else:
            action_dim = 1
            self.actor = Actor_discrete(env, observation_dim, action_dim, args.hidden_dim)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=3e-5)
        
        # Q1, Q2, Q1_target, Q2_target
        self.critic1 = Critic(observation_dim, action_dim, args.hidden_dim)
        self.critic2 = Critic(observation_dim, action_dim, args.hidden_dim)
        self.critic1_target = Critic(observation_dim, action_dim, args.hidden_dim)
        self.critic2_target = Critic(observation_dim, action_dim, args.hidden_dim)

        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=args.lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=args.lr)
        self.critic1_target_optimizer = Adam(self.critic1_target.parameters(), lr=args.lr)
        self.critic2_target_optimizer = Adam(self.critic2_target.parameters(), lr=args.lr)

        # Alpha
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = Adam([self.log_alpha], lr=args.lr)

        # CQL_alpha
        self.cql_alpha_log = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = Adam([self.cql_alpha_log], lr=self.args.lr)

        self.logger = {
            'total_timesteps' : [],
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
        for idx in range(len(self.dataset['observations'])):
            state = self.dataset['observations'][idx]
            action = self.dataset['actions'][idx]
            reward = self.dataset['rewards'][idx]
            next_state = self.dataset['next_observations'][idx]
            done = self.dataset['terminals'][idx]
            replay_buffer.push(state, action, reward, next_state, done)

        for _ in range(self.args.num_episodes):
            if len(replay_buffer) > self.args.batch_size:
                for _ in range(self.args.updates):
                    batch = replay_buffer.sample(self.args.batch_size)
                    self.update(batch, gradient_steps)
                    gradient_steps += 1
            
            self.logger['total_timesteps'].append(total_timesteps)
            total_timesteps += 1

            if self.args.save and total_timesteps % self.args.save_freq == 0:
                print("New model saved!!!!!!!!!!!!")
                torch.save(self.critic1.state_dict(), f'./models/critic1_{self.args.env}.pt')
                torch.save(self.critic2.state_dict(), f'./models/critic2_{self.args.env}.pt')
                torch.save(self.critic1_target.state_dict(), f'./models/critic1_target_{self.args.env}.pt')
                torch.save(self.critic2_target.state_dict(), f'./models/critic2_target_{self.args.env}.pt')
                torch.save(self.actor.state_dict(), f'./models/actor_{self.args.env}.pt')
                torch.save(self.alpha, f'./models/alpha_{self.args.env}.pt')
                torch.save(self.cql_alpha, f'./models/cql_alpha{self.args.env}.pt')
            
            if total_timesteps % self.args.logging_freq == 0:
                self.logging()

            if total_timesteps % self.args.eval_freq == 0:
                self.evaluation()
            
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

        """
        Compute critic loss : std_bellman_error_loss, cql_loss(-cql_alpha_loss)
        """
        # Compute std_bellman_error_loss
        # next states come from replay buffer and next actions come from current policy
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_target = self.critic1_target(next_state, next_action)
            q2_target = self.critic2_target(next_state, next_action)
            q_target = reward + (1 - done) * self.args.gamma * (torch.min(q1_target, q2_target) - self.alpha * next_log_prob)

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)

        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        
        std_bellman_error_loss = q1_loss + q2_loss

        # Compute CQL loss
        random_sampled_actions = torch.tensor([])
        for i in range(action.shape[-1]):
            temp = torch.FloatTensor(state.shape[0] * self.num_samples, 1).uniform_(self.env.action_space.low[i], self.env.action_space.high[i])
            random_sampled_actions = torch.cat((random_sampled_actions, temp), dim=1)

        repeated_states = state.repeat(self.num_samples, 1)
        repeated_next_states = next_state.repeat(self.num_samples, 1)

        with torch.no_grad():
            repeated_actions, repeated_log_probs = self.actor.sample(repeated_states)
            repeated_next_actions, repeated_next_log_probs = self.actor.sample(repeated_next_states)

        q1_from_random = self.critic1(repeated_states, random_sampled_actions)
        q2_from_random = self.critic2(repeated_states, random_sampled_actions)
        rand_density = np.log(0.5 ** repeated_actions.shape[-1])
        
        q1_from_curr_actions = self.critic1(repeated_states, repeated_actions)
        q2_from_curr_actions = self.critic2(repeated_states, repeated_actions)

        q1_from_next_actions = self.critic1(repeated_states, repeated_next_actions)
        q2_from_next_actions = self.critic2(repeated_states, repeated_next_actions)

        cql_q1 = torch.cat([q1_from_random - rand_density, q1_from_curr_actions - repeated_log_probs, q1_from_next_actions - repeated_next_log_probs], dim=1)
        cql_q2 = torch.cat([q2_from_random - rand_density, q2_from_curr_actions - repeated_log_probs, q2_from_next_actions - repeated_next_log_probs], dim=1)
        
        logsumexp_q1 = torch.logsumexp(cql_q1, dim=1)
        logsumexp_q2 = torch.logsumexp(cql_q2, dim=1)

        repeated_q1 = q1.repeat(self.num_samples, 1)
        repeated_q2 = q2.repeat(self.num_samples, 1)

        self.cql_alpha = torch.clamp(torch.exp(self.cql_alpha_log), min=0.0, max=1e6)
        cql_loss1 = self.cql_alpha * self.args.cql_tau * ((logsumexp_q1 - repeated_q1).mean() - 1)
        cql_loss2 = self.cql_alpha * self.args.cql_tau * ((logsumexp_q2 - repeated_q2).mean() - 1)
        cql_alpha_loss = -(cql_loss1 + cql_loss2) * 0.5  # For maximizing alpha, add -

        self.cql_alpha_optimizer.zero_grad()
        cql_alpha_loss.backward(retain_graph=True)
        self.cql_alpha_optimizer.step()

        critic_loss = std_bellman_error_loss + cql_loss1 + cql_loss2

        """
        Compute actor loss
        """
        action_from_current, log_prob_from_current = self.actor.sample(state)
        q1_from_current = self.critic1(state, action_from_current)
        q2_from_current = self.critic2(state, action_from_current)
        q_from_current = torch.min(q1_from_current, q2_from_current)
        actor_loss = -(q_from_current - self.alpha * log_prob_from_current).mean()

        """
        Compute alpha loss (SAC)
        """
        # Compute alpha loss
        alpha_loss = -(self.log_alpha * (log_prob_from_current.detach() + self.target_entropy)).mean()
        

        """
        Parameter updates
        """
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Append log
        self.logger['critic_loss'].append(critic_loss.item())
        self.logger['actor_loss'].append(actor_loss.item())
        self.logger['alpha_loss'].append(alpha_loss.item())
        self.logger['q_target'].append(torch.sum(q_target)/len(q_target))
        self.logger['q1'].append(torch.sum(q1)/len(q1))
        self.logger['q2'].append(torch.sum(q2)/len(q2))
        if self.args.continuous_space:
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
        if self.args.continuous_space:
            return action.detach().cpu().numpy()[0]
        else:
            return torch.argmax(action).detach().cpu().numpy()

    def evaluation(self):
        print('\n')
        print('#'*50)
        print('Start Evaluation!!!!!!!!!!!')

        with torch.no_grad():
            state = self.env.reset()
            done = False
            episode_steps = 0
            episode_reward = 0
            while not done:
                if self.args.render:
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state

                if episode_steps > self.args.max_episode_len:
                    done = True
                    break

            print('\n')
            print(f"Epdisode steps : {episode_steps}")
            print(f"Epdisode reward : {episode_reward}")
            print('#'*50)
            print('\n\n\n')

    def logging(self):
        total_timesteps = self.logger['total_timesteps'][0]
        avg_critic_loss = sum(self.logger['critic_loss']) / (len(self.logger['critic_loss']) + 1e-7)
        avg_actor_loss = sum(self.logger['actor_loss']) / (len(self.logger['actor_loss']) + 1e-7)
        avg_alpha_loss = sum(self.logger['alpha_loss']) / (len(self.logger['alpha_loss']) + 1e-7)
        avg_q_target = sum(self.logger['q_target']) / (len(self.logger['q_target']) + 1e-7)
        avg_q1 = sum(self.logger['q1']) / (len(self.logger['q1']) + 1e-7)
        avg_q2 = sum(self.logger['q2']) / (len(self.logger['q2']) + 1e-7)
        avg_log_prob = sum(self.logger['log_prob']) / (len(self.logger['log_prob']) + 1e-7)
        
        print('#'*50)
        print(f"Total timesteps : {total_timesteps}")
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

    def test(self):        
        self.actor.load_state_dict(torch.load(f"{self.args.models_path}/actor_{self.args.env}.pt"))

        for _ in range(10):
            done = False
            state = self.env.reset()
            episode_steps = 0
            episode_reward = 0

            while not done:
                self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                    
                if episode_steps >= self.args.max_episode_len:
                    done = True
                    break

            print('#'*50)
            print(f"Epdisode steps : {episode_steps}")
            print(f"Epdisode reward : {episode_reward}")
            print('#'*50)
            print('\n')