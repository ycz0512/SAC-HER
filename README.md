# Soft Actor-Critic with Hindsight Experience Replay (SAC-HER)
This framework provides an implementation of SAC-HER algorithm which enables the training of goal-conditioned policies to perform goal-based robotic tasks on standard *OpenAI Gym* environments.

SAC-HER achieves competitive performances with corresponding *gym* baselines in terms of the sample efficiency and asymtotic performance.
(in some tasks even better)

## Getting Started

### Prerequisites

1. To get everything installed correctly, you will first need to clone [rllab](https://github.com/rll/rllab), and have its path added to your PYTHONPATH environment variable.

```
cd <installation_path_of_rllab>
git clone https://github.com/rll/rllab.git
cd rllab
git checkout b3a28992eca103cab3cb58363dd7a4bb07f250a0
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

2. Download [mujoco](https://www.roboti.us/index.html) (mjpro150 linux) and copy several files to rllab path: 

```
mkdir <installation_path_of_rllab>/rllab/vendor/mujoco
cp <installation_path_of_mujoco>/mjpro150/bin/libmujoco150.so <installation_path_of_rllab>/rllab/vendor/mujoco
cp <installation_path_of_mujoco>/mjpro150/bin/libglfw.so.3 <installation_path_of_rllab>/rllab/vendor/mujoco
```

3. Copy your mujoco license key (mjkey.txt) to rllab path:

`cp <mujoco_key_folder>/mjkey.txt <installation_path_of_rllab>/rllab/vendor/mujoco`

### Install Requirements

`pip install -r requirements.txt`

and everything should be good to go.

## Training Models

### SAC-HER

To train a policy using SAC-HER, run the following command:

`python examples/run_sac_her.py --env=FetchPush --exp_name=sac-her`

- `FetchPush` can be replaced with other goal-based roboric environments.
- `--exp_name` specifies the experiment name. If you remove the flag, the experiment name will be the current timestamp by default.
- The log(.csv) and model(.pkl) will be saved to the `./data` directory by default. But the output directory can also be specified with `--log_dir=[log-directory]`.

### SAC

To train a policy using SAC, run the following command:

`python examples/run_sac_goal.py --env=FetchPush --exp_name=sac-goal`

This will simply run SAC under the goal-aware observation space (observation space augmented with goal informations).

`run_sac_her.py` and `run_sac_goal.py` contains several different environments.
For more information about the agents and configurations, run the scripts with `--help` flag. For example:
```
python examples/run_sac_her.py --help
usage: run_sac_her.py [-h]
                      [--env {FetchReach,FetchPush,FetchSlide,FetchPickAndPlace,HandReach,HandBlock,HandEgg,HandPen}]
                      [--exp_name EXP_NAME] [--log_dir LOG_DIR]
```

## Visualizing Models

To simulate the trained policies, run:

`python examples/visualize.py <model-directory> --max-path-length=50`

- `<model-directory>` specifies the directory of `.pkl` file of the trained model.
- `--max-path-length` specifies the maximum environment steps. If you remove the flag, this will be 20 by default. Note that 1 environment step = 20 simulation steps.
