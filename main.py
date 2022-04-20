import argparse
from src.sac import SAC
import gym


parser = argparse.ArgumentParser(description="SAC training.")
parser.add_argument('--buffer_size', dest='buffer_size', default=int(1e6), type=int)
parser.add_argument('--hidden_dim', nargs='+', dest='hidden_dim', default=[256], type=int)
parser.add_argument('--lr', dest='lr', default=3e-4, type=float)
parser.add_argument('--gamma', dest='gamma', default=0.99, type=float)
parser.add_argument('--alpha', dest='alpha', default=0.05, type=float)
parser.add_argument('--tau', dest='tau', default=0.005, type=float)
parser.add_argument('--batch_size', dest='batch_size', default=256, type=int)
parser.add_argument('--num_episodes', dest='num_episodes', default=int(3e7), type=int)
parser.add_argument('--timesteps', dest='timesteps', default=int(3e7), type=int)
parser.add_argument('--target_update', dest='target_update', default=1, type=int)
parser.add_argument('--updates', dest='updates', default=1, type=int)
parser.add_argument('--env', dest='env', default='Pendulum-v1', type=str)
parser.add_argument('--render', dest='render', default=False, type=bool)
parser.add_argument('--render_period', dest='render_period', default='50', type=int)
parser.add_argument('--logging_period', dest='logging_period', default='1', type=int)
parser.add_argument('--hard_target', dest='hard_target', default=False, type=bool)

args = parser.parse_args()

if args.hard_target:
    print("Hard target updating!!!!!!")
    args.target_update = 1000
else:
    print("Soft target updating!!!!!!")

env = gym.make(args.env)
agent = SAC(env, args)
agent.train()