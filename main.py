import argparse
import pickle
from src.sac import SAC
from src.cql_sac import CQL_sac
import gym
import d4rl


parser = argparse.ArgumentParser(description="SAC training.")
parser.add_argument('--algo', dest='algo', default='cql', type=str)
parser.add_argument('--gamma', dest='gamma', default=0.99, type=float)
parser.add_argument('--alpha', dest='alpha', default=0.05, type=float)
parser.add_argument('--tau', dest='tau', default=0.005, type=float)
parser.add_argument('--cql_alpha', dest='cql_alpha', default=5., type=float)
parser.add_argument('--cql_tau', dest='cql_tau', default=10., type=float)
parser.add_argument('--cql_scaling', dest='cql_scaling', default=5., type=float)

parser.add_argument('--batch_size', dest='batch_size', default=256, type=int)
parser.add_argument('--hidden_dim', nargs='+', dest='hidden_dim', default=[256, 256], type=int)
parser.add_argument('--lr', dest='lr', default=3e-4, type=float)
parser.add_argument('--buffer_size', dest='buffer_size', default=int(1e6), type=int)

parser.add_argument('--timesteps', dest='timesteps', default=int(3e7), type=int)
parser.add_argument('--num_episodes', dest='num_episodes', default=int(3e7), type=int)
parser.add_argument('--max_episode_len', dest='max_episode_len', default=1000, type=int)
parser.add_argument('--target_update', dest='target_update', default=1, type=int)
parser.add_argument('--updates', dest='updates', default=1, type=int)
parser.add_argument('--env', dest='env', default='hopper-expert-v0', type=str)

parser.add_argument('--hard_target', dest='hard_target', default=False, type=bool)
parser.add_argument('--continuous_space', dest='continuous_space', default=True, type=bool)
parser.add_argument('--mode', dest='mode', default='train', type=str)
parser.add_argument('--save', dest='save', default=False, type=bool)
parser.add_argument('--render', dest='render', default=False, type=bool)

parser.add_argument('--render_freq', dest='render_freq', default=50, type=int)
parser.add_argument('--logging_freq', dest='logging_freq', default=100, type=int)
parser.add_argument('--save_freq', dest='save_freq', default=1000, type=int)
parser.add_argument('--eval_freq', dest='eval_freq', default=1000, type=int)
parser.add_argument('--models_path', dest='models_path', default='./models', type=str)
parser.add_argument('--dataset', dest='dataset', default='d4rl', type=str)


args = parser.parse_args()

if args.hard_target:
    print("Hard target updating!!!!!!")
    args.target_update = 1000
else:
    print("Soft target updating!!!!!!")

env = gym.make(args.env)

if isinstance(env.action_space, gym.spaces.discrete.Discrete):
    args.continuous_space = False

if args.algo == 'sac':
    agent = SAC(env, args)
elif args.algo == 'cql':
    if args.dataset == 'd4rl':
        dataset = d4rl.qlearning_dataset(env)
    else:
        with open(f'./{args.env}.pkl', 'rb') as doo:
            dataset = pickle.load(doo)
    agent = CQL_sac(env, dataset, args)

if args.mode == 'train':
    agent.train()
elif args.mode == 'test':
    agent.test()
elif args.algo == 'sac' and args.mode == 'generate':
    agent.generate_dataset()