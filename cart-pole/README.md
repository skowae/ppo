# Proximal Policy Optimization (PPO)
Proximal Policy Optimization is a reinfocement learning algorithm proposed by [Schulman et al.](https://arxiv.org/abs/1707.06347).  The key idea is that it uses to parametermizations to define a policy for selecting actions and approximating the value function.  The model that selects actions is refered to as the actor, and the model that estimates the value function is called the critic.

In order to implement the PPO algorithm for the cart-pole problem we will use several abstractions. 

## Actor and Critic Networks
The actor model will generate the policy for our agent to solve a number of problems.  The network will be fairly simple, featuring two fully connected layers and a single action head to produce softmax probabilities of the agent taking each action in the action space

## Trajectory Structure
It will be useful to store the trajectories in an object for the models as we train.  The trajetories will contain: state, action, action_porbs, log_probs, values, rewards, and dones.

## Agent
The actor and critic networks and the trajectories will be stored in an Agent class along with other relevant functions

## Environment and Training
Now that we have our models designed and the training functions written we need to set up our environment and begin training.

## Util Functions
We need some helper functions in order to make training out PPO models easier

*   select_action: selects an action based on the actor and critic models and the current state
*   calculate_advantages: calculates the advantages for a trajectory
*   ppo_update: Optimizes the PPO models
*  train_ppo: A PPO training function
