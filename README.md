# CQL
Implementation of [Conservative Q-Learning](https://arxiv.org/abs/2006.04779) and [Soft Actor-Critic](https://arxiv.org/abs/1812.05905) algorithm in PyTorch.

## Requirements
- python 3.6+
- mujoco

## Installation Instructions
First, install [D4RL benchmark](https://github.com/rail-berkeley/d4rl) repository by following its installation instructions.

Create a virtual environment and install all required packages.
```
cd cql_impl
pip install -r requirements.txt
pip install -e .
```

## Example Commands
### SAC
To train a SAC model for the environment, run:
```
python main.py --algo sac --env <env e.g. Pendulum-v1> --logging_freq 100 --save True --render False
```
For testing a trained SAC agent on the environment, run:
```
python main.py --algo sac --env <env e.g. Pendulum-v1> --mode test
```
For generating dataset with trained SAC agent, run:
```
python main.py --algo sac --env <env e.g. Pendulum-v1> -- mode generate
```
**Note!** You can skip dataset generating if you use d4rl datset.

### CQL
To train a CQL model for the environment, run:
```
python main.py --env <env e.g. hopper-expert-v0> --save True --cql_scaling (5.0 or 10.0) --cql_tau (5.0 or 10.0)
```
If you want to use your own dataset generated in the above manner, you can add `--dataset mine`. 
**Note!** You can try other valeus in `cql_scaling` and `cql_tau`. But, if you don't choose the value carefully, the `cql_loss` may become extremely large. 
For testing a trained CQL agent on the environment, run:
```
python main.py --env <env e.g. hopper-expert-v0> --mode test
```

You can see the results by using `Tensorboard`
```
tensorboard --logidr ./results/<env>/
```

## Results
### Average Return graph
<p align="center">
<img src="https://user-images.githubusercontent.com/48791681/165443768-07e513ed-b15f-476c-8150-795590ffd9a6.png" width="600">
</p>
</img>

### Trained Agent
<p align="center">
<gif src="https://user-images.githubusercontent.com/48791681/165444366-6ea6d36f-4591-4858-9d45-8c19318fcb26.gif" width="600">
</p>
</gif>
![hopper](https://user-images.githubusercontent.com/48791681/165444366-6ea6d36f-4591-4858-9d45-8c19318fcb26.gif)
