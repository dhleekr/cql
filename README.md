# cql_imql
## Requirements
- python 3.6+
- mujoco

## Installation Instructions
First, install [D4RL benchmark](https://github.com/rail-berkeley/d4rl) repository by following its installation instructions.

Create a virtual environment and install all required packages.
'''
cd cql_impl
pip install -r requirements.txt
pip install -e .
'''

## Example Commands
### SAC
To train a SAC model for the environment, run:
'''
python main.py --algo sac --env <env e.g. Pendulum-v1> --logging_freq 100 --save True --render False
'''
For testing a trained SAC agent on the environment, run:
'''
python main.py --algo sac --env <env e.g. Pendulum-v1> --mode test
'''
For generating dataset with trained SAC agent, run:
'''
python main.py --algo sac --env <env e.g. Pendulum-v1> -- mode generate
'''
**Note** You can skip dataset generating if you use d4rl datset.

### CQL
To train a CQL model for the environment, run:
'''
python main.py --env <env e.g. hopper-expert-v0> --save True --cql_scaling (5.0 or 10.0) --cql_tau (5.0 or 10.0)
'''
**Note** If you want to use your own dataset generated in the above manner, you can add '--dataset mine'
For testing a trained CQL agent on the environment, run:
'''
python main.py --env <env e.g. hopper-expert-v0> --mode test
'''