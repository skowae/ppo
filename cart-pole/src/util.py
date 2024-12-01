from Trajectories import *
import torch
import torch.nn as nn
from datetime import datetime
import json
import os
import numpy as np

def select_action(actor, critic, state, trajectory):
  """
  Selects an action based on the output of the actor and critic models

  :param actor: The actor model
  :param critic: The critic model
  :param state: The current state
  :param trajectory: The memory object to store results
  :return: The action to take
  """
  # Run the actor model and extract an action from the probabilities
  action_probs = actor(state)
  action = action_probs.sample()

  # Run the critic model and extract a value
  value = critic(state)

  # Update the memory
  trajectory.states.append(state)
  trajectory.actions.append(action)
  trajectory.action_probs.append(action_probs)
  trajectory.log_probs.append(action_probs.log_prob(action).detach())
  trajectory.values.append(value.detach())

  # Return the action to take
  return action.item()

def calculate_advantages(trajectory, gamma=0.99, lmbda=0.95):
  """
  Calculates the advantages for a trajectory.  Calculates the advantages with 
  a generalized advantage estimation algorithm.

  :param trajectory: The memory object to store results
  :param gamma: The discount factor
  :param lmbda: The tradeoff parameter

  :return advantages: The advantages
  :return returns: The discoutned returns
  """
  # Extract the values, rewards, and dones from the trajectory
  values = torch.cat(trajectory.values)
  rewards = torch.cat(trajectory.rewards)
  dones = torch.cat(trajectory.dones)

  # Calculate the discounted rewards
  advantages = []
  gae = 0
  next_value = 0  # Value after the final step (0 if the episode ends)
  for t in reversed(range(len(rewards))):
    # Check if this is not the penultimate step
    if t < len(rewards) - 1:
      next_value = values[t + 1]
    
    # Calculate the delta value
    delta = rewards[t] + gamma*next_value*(1 - dones[t]) - values[t]
    # Calculate the gae
    gae = delta + gamma*lmbda*gae*(1 - dones[t])
    advantages.insert(0, gae)

  # convert the advantages to a tensor
  advantages = torch.tensor(advantages, dtype=torch.float32)

  # Calculate the returns
  returns = advantages + values.squeeze()

  # Return the advantages and returns
  return advantages.detach(), returns.detach()

def ppo_update(actor, critic, actor_optimizer, critic_optimizer, trajectories, 
               epochs, batch_size, clip_param, gamma, lmbda, entropy_coeff):
  """
  Runs a Proximal Policy Optimization update on the actor and critic models.  
  It will run a specified number of training epochs on the provided trajectories 
  divided into batches according to the batch size.

  :param actor: The actor model
  :param critic: The critic model
  :param actor_optimizer: The optimizer for the actor model
  :param critic_optimizer: The optimizer for the critic model
  :param trajectories: The trajectories to train on
  :param epochs: The number of training epochs
  :param batch_size: The training batch size
  :param clip_param: The clipping parameter for policy loss
  :param gamma: The discount factor
  :param lmbda: The tradeoff parameter
  """

  # Loss values for logging
  policy_losses = []
  value_losses = []
  entropies = []

  # Calculate the advantages and returns
  advantages, returns = calculate_advantages(trajectories, gamma, lmbda)

  # Normalize advantages
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  # Extract the states, actions, and log_probs from the trajectories
  states = torch.stack([s.detach() for s in trajectories.states])
  actions = torch.cat([action.unsqueeze(0).detach() for action in trajectories.actions])
  old_log_probs = torch.cat([log_prob.unsqueeze(0).detach() for log_prob in trajectories.log_probs])

  # Loop for the number of epochs
  for _ in range(epochs):
    # Loop over the trajectories in batches
    for batch_start in range(0, len(states), batch_size):
      # Extract the batch
      batch_end = batch_start + batch_size
      state_batch = states[batch_start:batch_end]
      action_batch = actions[batch_start:batch_end]
      old_log_probs_batch = old_log_probs[batch_start:batch_end]
      advantage_batch = advantages[batch_start:batch_end]
      return_batch = returns[batch_start:batch_end]

      ## Actor
      # Extract the current policy action_probs, log_probs, and entropy
      action_probs = actor(state_batch)
      log_probs = action_probs.log_prob(action_batch)
      entropy = action_probs.entropy().mean()
      entropy = 0
      entropies.append(entropy)
      # Calculate the ratio of probs
      ratio = torch.exp(log_probs - old_log_probs_batch)
      # Calculate surr1
      surr1 = ratio*advantage_batch
      # Calculate surr2
      surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)*advantage_batch
      # Calculate the policy loss
      policy_loss = -torch.min(surr1, surr2).mean() - entropy_coeff*entropy
      policy_losses.append(policy_loss.item())
      # Gradient decent
      actor_optimizer.zero_grad()
      policy_loss.backward()
      actor_optimizer.step()

      ## Critic
      # Extract the current value estimate from the critic 
      values = critic(state_batch)
      # Calculate the value loss
      value_loss = nn.MSELoss()(values, return_batch.unsqueeze(1))
      value_losses.append(value_loss.item())
      # Gradient decent
      critic_optimizer.zero_grad()
      value_loss.backward()
      critic_optimizer.step()
    
    # Return the average losses and entropy and average return
    return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies), returns.mean()

def train_ppo(env, actor, critic, actor_optimizer, critic_optimizer, 
              episodes_per_update, num_episodes, epochs, log_dir, batch_size=64, 
              clip_param=0.2, gamma=0.99, lmbda=0.95, entropy_coeff=0.01):
  
  assert os.path.isdir(os.path.join(log_dir,'logs')), f"Directory {log_dir}/logs/ does not exist. Please create it before running the program."
  # Load or initialize the log data
  log_file = f"{log_dir}/logs/training_logs.json"
  if os.path.exists(log_file):
    with open(log_file, "r") as f:
        log_data = json.load(f)
  else:
    log_data = []

  # Create the trajectories object
  trajectories = Trajectories(batch_size=64)
  episode_count = 0
  update_count = 0
  episode_rewards = []

  # Reset the environment
  state = env.reset()
  state = torch.from_numpy(state).float()
  total_reward = 0
  episode_start = 0

  while episode_count < num_episodes:
    # Select action will store state and action vals in trajectories
    action = select_action(actor, critic, state, trajectories)
    next_state, reward, done, _ = env.step(action)

    # Update the trajectories
    trajectories.rewards.append(torch.tensor([reward], dtype=torch.float32))
    trajectories.dones.append(torch.tensor([1.0 if done else 0.0], dtype=torch.float32))
    total_reward += reward
    state = torch.from_numpy(next_state).float()

    if done:
      # Append the total_reward
      episode_rewards.append(total_reward)
      state = env.reset()
      state = torch.from_numpy(state).float()
      total_reward = 0
      episode_count += 1
      episode_start = len(trajectories.states)

      # Perform PPO update after completing episodes_per_update
      if episode_count % episodes_per_update == 0:
        initial_policy_params = {name: param.clone() for name, param in actor.named_parameters()}
        initial_critic_params = {name: param.clone() for name, param in critic.named_parameters()}

        policy_loss, value_loss, entropy, avg_return = ppo_update(actor, critic, 
                                                                  actor_optimizer, 
                                                                  critic_optimizer, 
                                                                  trajectories, 
                                                                  epochs, 
                                                                  batch_size, 
                                                                  clip_param, 
                                                                  gamma, lmbda, 
                                                                  entropy_coeff)
        print(f"Episode {episode_count}, Total Reward: {episode_rewards[-1]}, Avg Return {avg_return}, Policy Loss: {policy_loss}, Value Loss: {value_loss}")
        # Check updates
        for name, param in actor.named_parameters():
            assert (not torch.equal(initial_policy_params[name], param)), f'{param} was not updated'

        for name, param in critic.named_parameters():
            assert (not torch.equal(initial_critic_params[name], param)), f'{param} was not updated'

        update_count += 1
        log_entry = {
          "Episode": episode_count,
          "average_return": avg_return.item(),
          "policy_loss": policy_loss.item(),
          "value_loss": value_loss.item(),
          "entropy": entropy,
          "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        log_data.append(log_entry)

        with open(log_file, "w") as f:
            f.write(json.dumps(log_data))

        # Save the updated weights
        assert os.path.isdir(os.path.join(log_dir,'models')), f"Directory {log_dir}/models/ does not exist. Please create it before running the program."
        epoch_dir = f"{log_dir}/models/update_{update_count}"
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(actor.state_dict(), f"{log_dir}/models/update_{update_count}/actor.pth")
        torch.save(critic.state_dict(), f"{log_dir}/models/update_{update_count}/critic.pth")

        trajectories.clear()
        
  return log_data

def load_logs(log_file):
  """
  Load logs from a JSON file.

  :param log_file: Path to the JSON log file
  :return: List of log entries
  """
  with open(log_file, "r") as f:
      return json.load(f)

import matplotlib.pyplot as plt
import numpy as np

def smooth_plot(log_data, metric_key, window_size=10, scenario_name="", ylabel="Value"):
  """
  Generate a smoothed plot where each y value is an average over a rolling window of epochs.
  The x-axis reflects the true epoch numbers.

  :param log_data: List of log entries
  :param metric_key: The key in the log entries to plot (e.g., "average_reward", "policy_loss")
  :param window_size: Number of epochs to average over for smoothing
  :param scenario_name: Title of the plot for context
  :param ylabel: Y-axis label for the plot
  """
  # Extract data, ignoring entries where the metric is NaN
  epochs = [entry["episode"] for entry in log_data if not np.isnan(entry[metric_key])]
  metrics = [entry[metric_key] for entry in log_data if not np.isnan(entry[metric_key])]

  # Calculate rolling averages
  smoothed_metrics = np.convolve(metrics, np.ones(window_size) / window_size, mode='valid')

  # Align epochs to match the smoothed data
  aligned_epochs = epochs[window_size - 1:]

  # Plot the smoothed data
  plt.figure(figsize=(10, 6))
  plt.plot(epochs, metrics, alpha=0.3, label=f"Raw {metric_key.replace('_', ' ').title()}")
  plt.plot(aligned_epochs, smoothed_metrics, label=f"Smoothed ({window_size} epochs)")
  plt.title(f"{metric_key.replace('_', ' ').title()} Over Training: {scenario_name}")
  plt.xlabel("Episode")
  plt.ylabel(ylabel)
  plt.legend()
  plt.grid()
  plt.show()

def visualize_logs(log_file, scenario_name):
  """
  Generate all graphs from the logs.

  :param log_dir: Directory containing the JSON logs
  :param scenario_name: Name of the scenario for the plot titles
  """
  # log_file = os.path.join(log_dir, "training_logs.json")
  if not os.path.exists(log_file):
      print(f"No logs found at {log_file}")
      return

  log_data = load_logs(log_file)

  # Generate all graphs
  smooth_plot(log_data, "score", window_size=10, scenario_name=scenario_name, ylabel="Score")
  smooth_plot(log_data, "avg_score", window_size=10, scenario_name=scenario_name, ylabel="Average Score")
  smooth_plot(log_data, "avg_value_loss", window_size=10, scenario_name=scenario_name, ylabel="Average Value Loss")





