[//]: # (Image References)

[image1]: reacher-agent.gif "Trained Agent"
[image2]: random-reacher.gif "Random Agent"

### Introduction

In this project, we will train an agent to to solve the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#reacher) 
environment.  

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that 
the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target 
location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of 
the target. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the 
action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 
consecutive episodes.

### Getting Started

1. Download the environment from one of the links below. In this repo I am using the Mac OSX environment, you only need 
   to select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

   (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)
   if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

   (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), 
   then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) 
   to obtain the environment.

2. Set `env_file` in the `settings.toml` file to the path where your environment file is stored, I'm using the
   `environment` dir.

3. Install the dependencies:
    - cd to the navigation directory.
    - activate your virtualenv.
    - run: `pip install -r requirements.txt` in your shell.

### Instructions

There are three options currently available to run `contrinuous-control.py`:

1. Run `python contrinuous-control.py -r` to run the reacher environment with an agent that selects actions randomly.

   ![Random Agent][image2]

   Not the best agent for this task but maybe a good one for robotic dancing though!


2. Run `python contrinuous-control.py -t` to train an agent to solve the Reacher environment while collecting experience
   in it.
   Hyperparameters are specified in `settings.toml`, feel free to tune them to see if you can get better results! Also,
   you can change `save_freq` which controls how often the agent weights are stored in the output dir.


3. Run `python contrinuous-control.py -c` and pass the path to the dir where an agent's weights are stored, for example
   `output/2021-05-30_reach-ppo/2021-05-30_09-13-25-reach-ppo_s24` to use that trained agent to explore the Reacher
   environment.