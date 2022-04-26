from .networks import Actor_continuous, Actor_discrete,Critic
from collections import deque, namedtuple
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter



Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

class ReplayBuffer:
    def __init__(self, buffer_size=2e6):
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
            self.actor = Actor_continuous(env, observation_dim, action_dim, args.hidden_dim).to(device)
        else:
            action_dim = 1
            self.actor = Actor_discrete(env, observation_dim, action_dim, args.hidden_dim).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=3e-5)
        
        # Q1, Q2, Q1_target, Q2_target
        self.critic1 = Critic(observation_dim, action_dim, args.hidden_dim).to(device)
        self.critic2 = Critic(observation_dim, action_dim, args.hidden_dim).to(device)
        self.critic1_target = Critic(observation_dim, action_dim, args.hidden_dim).to(device)
        self.critic2_target = Critic(observation_dim, action_dim, args.hidden_dim).to(device)

        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=args.lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=args.lr)

        # Alpha
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = Adam([self.log_alpha], lr=args.lr)

        # CQL_alpha
        self.cql_alpha_log = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = Adam([self.cql_alpha_log], lr=self.args.lr)

        self.logger = {
            'total_timesteps' : [],
            'actor_loss' : [],
            'critic_loss' : [],
            'q1_loss' : [],
            'q2_loss' : [],
            'cql_loss1' : [],
            'cql_loss2' : [],
            'alpha_loss' : [],
            'cql_alpha_loss' : [],
            'q_target' : [],
            'q1' : [],
            'q2' : [],
            'q1_from_random' : [],
            'q2_from_random' : [],
            'q1_from_curr_actions' : [],
            'q2_from_curr_actions' : [],
            'q1_from_next_actions' : [],
            'q2_from_next_actions' : [],
            'logsumexp_q1' : [],
            'logsumexp_q2' : [],
            'q1_diff' : [],
            'q2_diff' : [],
            'alpha' : [],
            'cql_alpha' : [],
            'log_prob' : []
        }
        self.writer = SummaryWriter(f"./results/{self.args.env}")

        print("Initializing...")

    def train(self):
        print("Training...")
        replay_buffer = ReplayBuffer(self.args.buffer_size)
        self.total_timesteps = 0
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
            
            if self.args.save and self.total_timesteps % self.args.save_freq == 0:
                print("New model saved!!!!!!!!!!!!")
                torch.save(self.critic1.state_dict(), f'./models/critic1_{self.args.env}.pt')
                torch.save(self.critic2.state_dict(), f'./models/critic2_{self.args.env}.pt')
                torch.save(self.critic1_target.state_dict(), f'./models/critic1_target_{self.args.env}.pt')
                torch.save(self.critic2_target.state_dict(), f'./models/critic2_target_{self.args.env}.pt')
                torch.save(self.actor.state_dict(), f'./models/actor_{self.args.env}.pt')
                torch.save(self.alpha, f'./models/alpha_{self.args.env}.pt')
                torch.save(self.cql_alpha, f'./models/cql_alpha_{self.args.env}.pt')
            
            if self.total_timesteps % self.args.logging_freq == 0:
                self.logger['total_timesteps'].append(self.total_timesteps) 
                self.logging()

            if self.total_timesteps % self.args.eval_freq == 0:
                self.evaluation()
            
            if self.total_timesteps > self.args.timesteps:
                break
            
            self.total_timesteps += 1
            
        self.writer.flush()
        self.writer.close()
        print("Finished!!!!")
               
    def update(self, batch, gradient_steps):
        state, action, reward, next_state, done = batch
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1).to(device)
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.float).unsqueeze(1).to(device)
        
        
        """
        Compute alpha loss (SAC)
        """
        # Compute alpha loss
        action_from_current, log_prob_from_current = self.actor.sample(state)
        alpha_loss = -(self.log_alpha.to(device) * (log_prob_from_current + self.target_entropy).detach().to(device)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = torch.exp(self.log_alpha).to(device)
        
        """
        Compute actor loss
        """
        q1_from_current = self.critic1(state, action_from_current)
        q2_from_current = self.critic2(state, action_from_current)
        q_from_current = torch.min(q1_from_current, q2_from_current)
        actor_loss = -(q_from_current - self.alpha * log_prob_from_current).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        """
        Compute critic loss : std_bellman_error_loss, cql_loss(-cql_alpha_loss)
        """
        # Compute std_bellman_error_loss
        # next states come from replay buffer and next actions come from current policy
        next_action, next_log_prob = self.actor.sample(next_state)
        q1_target = self.critic1_target(next_state, next_action)
        q2_target = self.critic2_target(next_state, next_action)
        q_target = reward + (1 - done) * self.args.gamma * (torch.min(q1_target, q2_target).to(device) - self.alpha * next_log_prob)

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)

        q1_loss = F.mse_loss(q1, q_target.detach())
        q2_loss = F.mse_loss(q2, q_target.detach())
        
        # std_bellman_error_loss = q1_loss + q2_loss

        # Compute CQL loss
        random_sampled_actions = torch.tensor([], requires_grad=False).to(device)
        for i in range(action.shape[-1]):
            temp = torch.FloatTensor(state.shape[0] * self.num_samples, 1).uniform_(self.env.action_space.low[i], self.env.action_space.high[i])
            random_sampled_actions = torch.cat((random_sampled_actions.to(device), temp.to(device)), dim=1)

        repeated_states = state.unsqueeze(1).repeat(1, self.num_samples, 1).view(state.shape[0] * self.num_samples, state.shape[1])
        repeated_next_states = next_state.unsqueeze(1).repeat(1, self.num_samples, 1).view(next_state.shape[0] * self.num_samples, next_state.shape[1])

        repeated_actions, repeated_log_probs = self.actor.sample(repeated_states)
        repeated_next_actions, repeated_next_log_probs = self.actor.sample(repeated_next_states)
        repeated_actions, repeated_log_probs = repeated_actions.detach(), repeated_log_probs.detach()
        repeated_next_actions, repeated_next_log_probs = repeated_next_actions.detach(), repeated_next_log_probs.detach()

        q1_from_random = self.critic1(repeated_states, random_sampled_actions).view(state.shape[0], self.num_samples, 1)
        q2_from_random = self.critic2(repeated_states, random_sampled_actions).view(state.shape[0], self.num_samples, 1)
        rand_density = np.log(0.5 ** repeated_actions.shape[1])
        
        q1_from_curr_actions = self.critic1(repeated_states, repeated_actions).view(state.shape[0], self.num_samples, 1)
        q2_from_curr_actions = self.critic2(repeated_states, repeated_actions).view(state.shape[0], self.num_samples, 1)

        q1_from_next_actions = self.critic1(repeated_states, repeated_next_actions).view(state.shape[0], self.num_samples, 1)
        q2_from_next_actions = self.critic2(repeated_states, repeated_next_actions).view(state.shape[0], self.num_samples, 1)

        cql_q1 = torch.cat([q1_from_random - rand_density,
                            q1_from_curr_actions - repeated_log_probs.view(state.shape[0], self.num_samples, 1), 
                            q1_from_next_actions - repeated_next_log_probs.view(state.shape[0], self.num_samples, 1)], 
                            dim=1)
        cql_q2 = torch.cat([q2_from_random - rand_density, 
                            q2_from_curr_actions - repeated_log_probs.view(state.shape[0], self.num_samples, 1), 
                            q2_from_next_actions - repeated_next_log_probs.view(state.shape[0], self.num_samples, 1)], 
                            dim=1)
          
        logsumexp_q1 = torch.logsumexp(cql_q1, dim=1).to(device)
        logsumexp_q2 = torch.logsumexp(cql_q2, dim=1).to(device)

        q1_diff = (logsumexp_q1 - q1).mean() * self.args.cql_scaling
        q2_diff = (logsumexp_q2 - q2).mean() * self.args.cql_scaling

        self.cql_alpha = torch.clamp(torch.exp(self.cql_alpha_log), min=0.0, max=1e6).to(device)
        cql_loss1 = self.cql_alpha * (q1_diff - self.args.cql_tau) 
        cql_loss2 = self.cql_alpha * (q2_diff - self.args.cql_tau)
        cql_alpha_loss = -(cql_loss1 + cql_loss2) * 0.5 # For maximizing alpha, add -

        self.cql_alpha_optimizer.zero_grad()
        cql_alpha_loss.backward(retain_graph=True)
        self.cql_alpha_optimizer.step()

        critic1_loss = q1_loss + cql_loss1
        critic2_loss = q2_loss + cql_loss2
        critic_loss = critic1_loss + critic2_loss
      
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        critic2_loss.backward(retain_graph=True)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Append log
        self.logger['actor_loss'].append(actor_loss.item())
        self.logger['critic_loss'].append((critic1_loss + critic2_loss).item())
        self.logger['q1_loss'].append(q1_loss.item())
        self.logger['q2_loss'].append(q2_loss.item())
        self.logger['cql_loss1'].append(cql_loss1.item())
        self.logger['cql_loss2'].append(cql_loss2.item())
        self.logger['alpha_loss'].append(alpha_loss.item())
        self.logger['cql_alpha_loss'].append(cql_alpha_loss.item())
        self.logger['q_target'].append(q_target.mean().item())
        self.logger['q1'].append(q1.mean().item())
        self.logger['q2'].append(q2.mean().item())
        self.logger['q1_from_random'].append(q1_from_random.mean().item())
        self.logger['q2_from_random'].append(q2_from_random.mean().item())
        self.logger['q1_from_curr_actions'].append(q1_from_curr_actions.mean().item())
        self.logger['q2_from_curr_actions'].append(q2_from_curr_actions.mean().item())
        self.logger['q1_from_next_actions'].append(q1_from_next_actions.mean().item())
        self.logger['q2_from_next_actions'].append(q2_from_next_actions.mean().item())
        self.logger['logsumexp_q1'].append(logsumexp_q1.mean().item())
        self.logger['logsumexp_q2'].append(logsumexp_q2.mean().item())
        self.logger['q1_diff'].append(q1_diff.item())
        self.logger['q2_diff'].append(q2_diff.item())
        self.logger['alpha'].append(self.alpha.item())
        self.logger['cql_alpha'].append(self.cql_alpha.item())
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
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
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
            avg_episode_reward = 0
            total_episode_steps = 0
            for _ in range(10):
                # self.env.seed(random.randint(0, 1000))
                state = self.env.reset()
                done = False
                episode_steps = 0
                episode_reward = 0
                for _ in range(self.args.max_episode_len):
                    if self.args.render:
                        self.env.render()
                    action = self.get_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    
                    episode_reward += reward
                    episode_steps += 1
                    state = next_state

                    if done:
                        break

                avg_episode_reward += episode_reward
                total_episode_steps += episode_steps
            
            avg_episode_reward /= 10
            avg_episode_steps = total_episode_steps / 10

            print(f"Average epdisode reward : {avg_episode_reward}")
            print(f"Average epdisode steps : {avg_episode_steps}")
            print('#'*50)
            print('\n\n\n')

            self.writer.add_scalar("Average return", avg_episode_reward, self.total_timesteps)

    def logging(self):
        total_timesteps = self.logger['total_timesteps'][0]
        actor_loss = sum(self.logger['actor_loss']) / (len(self.logger['actor_loss']) + 1e-7)
        critic_loss = sum(self.logger['critic_loss']) / (len(self.logger['critic_loss']) + 1e-7)
        q1_loss = sum(self.logger['q1_loss']) / (len(self.logger['q1_loss']) + 1e-7)
        q2_loss = sum(self.logger['q2_loss']) / (len(self.logger['q2_loss']) + 1e-7)
        cql_loss1 = sum(self.logger['cql_loss1']) / (len(self.logger['cql_loss1']) + 1e-7)
        cql_loss2 = sum(self.logger['cql_loss2']) / (len(self.logger['cql_loss2']) + 1e-7)
        alpha_loss = sum(self.logger['alpha_loss']) / (len(self.logger['alpha_loss']) + 1e-7)
        cql_alpha_loss = sum(self.logger['cql_alpha_loss']) / (len(self.logger['cql_alpha_loss']) + 1e-7)
        q_target = sum(self.logger['q_target']) / (len(self.logger['q_target']) + 1e-7)
        q1 = sum(self.logger['q1']) / (len(self.logger['q1']) + 1e-7)
        q2 = sum(self.logger['q2']) / (len(self.logger['q2']) + 1e-7)
        q1_from_random = sum(self.logger['q1_from_random']) / (len(self.logger['q1_from_random']) + 1e-7)
        q2_from_random = sum(self.logger['q2_from_random']) / (len(self.logger['q2_from_random']) + 1e-7)
        q1_from_curr_actions = sum(self.logger['q1_from_curr_actions']) / (len(self.logger['q1_from_curr_actions']) + 1e-7)
        q2_from_curr_actions = sum(self.logger['q2_from_curr_actions']) / (len(self.logger['q2_from_curr_actions']) + 1e-7)
        q1_from_next_actions = sum(self.logger['q1_from_next_actions']) / (len(self.logger['q1_from_next_actions']) + 1e-7)
        q2_from_next_actions = sum(self.logger['q2_from_next_actions']) / (len(self.logger['q2_from_next_actions']) + 1e-7)
        logsumexp_q1 = sum(self.logger['logsumexp_q1']) / (len(self.logger['logsumexp_q1']) + 1e-7)
        logsumexp_q2 = sum(self.logger['logsumexp_q2']) / (len(self.logger['logsumexp_q2']) + 1e-7)
        q1_diff = sum(self.logger['q1_diff']) / (len(self.logger['q1_diff']) + 1e-7)
        q2_diff = sum(self.logger['q2_diff']) / (len(self.logger['q2_diff']) + 1e-7)
        alpha = sum(self.logger['alpha']) / (len(self.logger['alpha']) + 1e-7)
        cql_alpha = sum(self.logger['cql_alpha']) / (len(self.logger['cql_alpha']) + 1e-7)
        log_prob = sum(self.logger['log_prob']) / (len(self.logger['log_prob']) + 1e-7)
        
        print('#'*50)
        print(f"Total timesteps\t\t|\t{total_timesteps}")
        print(f"actor loss\t\t|\t{actor_loss}")
        print(f"critic loss\t\t|\t{critic_loss}")
        print(f"q1_loss\t\t\t|\t{q1_loss}")
        print(f"q2_loss\t\t\t|\t{q2_loss}")
        print(f"cql_loss1\t\t|\t{cql_loss1}")
        print(f"cql_loss2\t\t|\t{cql_loss2}")
        print(f"alpha_loss\t\t|\t{alpha_loss}")
        print(f"cql_alpha_loss\t\t|\t{cql_alpha_loss}")
        print(f"q_target :\t\t|\t{q_target}")
        print(f"q1\t\t\t|\t{q1}")
        print(f"q2\t\t\t|\t{q2}")
        print(f"q1_from_random\t\t|\t{q1_from_random}")
        print(f"q2_from_random\t\t|\t{q2_from_random}")
        print(f"q1_from_curr_actions\t|\t{q1_from_curr_actions}")
        print(f"q2_from_curr_actions\t|\t{q2_from_curr_actions}")
        print(f"q1_from_next_actions\t|\t{q1_from_next_actions}")
        print(f"q2_from_next_actions\t|\t{q2_from_next_actions}")
        print(f"logsumexp_q1\t\t|\t{logsumexp_q1}")
        print(f"logsumexp_q2\t\t|\t{logsumexp_q2}")
        print(f"q1_diff\t\t\t|\t{q1_diff}")
        print(f"q2_diff\t\t\t|\t{q2_diff}")
        print(f"alpha\t\t\t|\t{alpha}")
        print(f"cql_alpha\t\t|\t{cql_alpha}")
        print(f"log_prob\t\t|\t{log_prob}")
        print('#'*50)
        print('\n')

        for k in self.logger.keys():
            self.logger[k] = []

    def test(self):        
        self.actor.load_state_dict(torch.load(f"{self.args.models_path}/actor_{self.args.env}.pt", map_location=device))

        for _ in range(15):
            done = False
            self.env.seed(random.randint(0, 1000))
            state = self.env.reset()
            episode_steps = 0
            episode_reward = 0

            while True:
                self.env.render()
                action = self.get_action(state)
                # action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                state = next_state
                    
                if done or episode_steps >= self.args.max_episode_len:
                    break

            print('#'*50)
            print(f"Epdisode steps : {episode_steps}")
            print(f"Epdisode reward : {episode_reward}")
            print('#'*50)
            print('\n')