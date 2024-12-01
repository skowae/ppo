# This script will test various functions between the PPO code from git hub and my own

import torch
import numpy as np
import Trajectories
import ppo_torch
import Agent
import random


def make_states(state_dim, num_states):
    # Generate a list of arrays with random floats between -1 and 1
    states = [np.random.uniform(-1, 1, state_dim).astype(float) for _ in range(num_states)]
    
    # Generate a list of actions as integers 0 or 1
    actions = [random.randint(0, 1) for _ in range(num_states)]
    
    # Generate a list of log_probs as random floats
    log_probs = [random.uniform(-10, 0) for _ in range(num_states)]  # log_probs are typically negative
    
    # Generate a list of booleans for dones
    dones = [random.choice([True, False]) for _ in range(num_states)]
    
    # Generate a list of values as floats between 0 and 20
    values = [random.uniform(0, 20) for _ in range(num_states)]
    
    # Generate a list of rewards as ones
    rewards = [1 for _ in range(num_states)]

    return states, actions, log_probs, values, rewards, dones


def test_generate_batches():

    # Create a fake trajectory and memory
    states, actions, log_probs, values, rewards, dones = make_states(4, 3)

    # Define the trajectories and PPOMemory
    traj = Trajectories.Trajectories(1)
    ppo_mem = ppo_torch.PPOMemory(1)

    # Load the memory
    for i in range(len(states)):
        traj.update_trajectory(states[i], actions[i], log_probs[i], values[i], rewards[i], dones[i])
        ppo_mem.store_memory(states[i], actions[i], log_probs[i], values[i], rewards[i], dones[i])
    
    traj_states, traj_actions, traj_log_probs, traj_values, traj_rewards, traj_dones, traj_batch = traj.generate_batches()
    ppo_states, ppo_actions, ppo_probs, ppo_vals, ppo_rewards, ppo_dones, ppo_batch = ppo_mem.generate_batches()

    # Check if the states match
    states_match = True
    for i, traj_state in enumerate(traj_states):
        if not np.array_equal(traj_state, ppo_states[i]):
            states_match = False
    
    actions_match = np.array_equal(traj_actions, ppo_actions)
    log_probs_match = np.array_equal(traj_log_probs, ppo_probs)
    values_match = np.array_equal(traj_values, ppo_vals)
    rewards_match = np.array_equal(traj_rewards, ppo_rewards)
    dones_match = np.array_equal(traj_dones, ppo_dones)

    print("Generate Batch Test")
    print(f"    States match: {states_match}")
    print(f"    Actions match: {actions_match}") 
    print(f"    Probs match: {log_probs_match}")
    print(f"    Values match: {values_match}")
    print(f"    Rewards match: {rewards_match}") 
    print(f"    Dones match: {dones_match}")
    print(f"    Traj batch: {traj_batch}")
    print(f"    PPO batch {ppo_batch}")

def test_choose_action():
    # Generate a fake observation
    states, _, __, ___, ____, _____ = make_states(4, 3)

    # Select a single state
    state = states[0]

    # Create the different agents 
    skow_agent = Agent.Agent(action_dim=2, input_dim=4, gamma=0.99, alpha=3e-4, lmda=0.95, 
               clip=0.2, batch_size=64, epochs=10, log_dir='/tmp/ppo')
    
    ppo_agent = ppo_torch.Agent(n_actions=2, input_dims=(4,), gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10)

    # Select an action from each agent
    skow_action, skow_log_probs, skow_value = skow_agent.choose_action(state)
    ppo_action, ppo_probs, ppo_value = ppo_agent.choose_action(state)

    print("Test Choose Action")
    print(f"    Skow Action: {skow_action}, ppo_action: {ppo_action}")
    print(f"    Skow log_probs: {skow_log_probs}, ppo_probs: {ppo_probs}")
    print(f"    Skow Value: {skow_value}, ppo_value: {ppo_value}")

def test_remember():
    # Create a fake trajectory and memory
    states, actions, log_probs, values, rewards, dones = make_states(4, 3)

    # Create the different agents 
    skow_agent = Agent.Agent(action_dim=2, input_dim=4, gamma=0.99, alpha=3e-4, lmda=0.95, 
               clip=0.2, batch_size=5, epochs=4, log_dir='/tmp/ppo')
    
    ppo_agent = ppo_torch.Agent(n_actions=2, input_dims=(4,), gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=5, n_epochs=4)
    
    # make the agents remember
    skow_agent.remember(states[0], actions[0], log_probs[0], values[0], rewards[0], dones[0])
    ppo_agent.remember(states[0], actions[0], log_probs[0], values[0], rewards[0], dones[0])

    # Check if the states match
    states_match = True
    for i, traj_state in enumerate(skow_agent.trajectories.states):
        if not np.array_equal(traj_state, ppo_agent.memory.states[i]):
            states_match = False
    
    actions_match = np.array_equal(skow_agent.trajectories.actions, ppo_agent.memory.actions)
    log_probs_match = np.array_equal(skow_agent.trajectories.log_probs, ppo_agent.memory.probs)
    values_match = np.array_equal(skow_agent.trajectories.values, ppo_agent.memory.vals)
    rewards_match = np.array_equal(skow_agent.trajectories.rewards, ppo_agent.memory.rewards)
    dones_match = np.array_equal(skow_agent.trajectories.dones, ppo_agent.memory.dones)

    print("Generate Batch Test")
    print(f"    States match: {states_match}")
    print(f"    Actions match: {actions_match}") 
    print(f"    Probs match: {log_probs_match}")
    print(f"    Values match: {values_match}")
    print(f"    Rewards match: {rewards_match}") 
    print(f"    Dones match: {dones_match}")

def test_advantage():
    # Generate the fake data
    states, actions, log_probs, values, rewards, dones = make_states(state_dim=4, num_states=10)

    # Create the different agents 
    skow_agent = Agent.Agent(action_dim=2, input_dim=4, gamma=0.99, alpha=3e-4, lmda=0.95, 
               clip=0.2, batch_size=2, epochs=4, log_dir='/tmp/ppo')
    
    ppo_agent = ppo_torch.Agent(n_actions=2, input_dims=(4,), gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=2, n_epochs=4)
    
    # Have the agents remember 
    for i in range(10):
        skow_agent.remember(states[i], actions[i], log_probs[i], values[i], rewards[i], dones[i])
        ppo_agent.remember(states[i], actions[i], log_probs[i], values[i], rewards[i], dones[i])
    
    # Create the batches 
    state_array, action_array, log_probs_array, value_array, reward_array,\
      done_array, batches = skow_agent.trajectories.generate_batches()
    
    state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    ppo_agent.memory.generate_batches()
    
    skow_advantages = skow_agent.calculate_advantages(value_array, reward_array, done_array)
    ppo_advantages = ppo_agent.calculate_advantages(vals_arr, reward_arr, dones_arr)

    advantages_match = np.array_equal(skow_advantages, ppo_advantages)

    print("Advantages Test:")
    print(f"    Advantages match: {advantages_match}")

def test_loss_vals():
    # Generate two sets of data
    num_states = 64
    batch_size = 2
    states, actions, log_probs, values, rewards, dones = make_states(4, num_states)

    # Create the agents
    skow_agent = Agent.Agent(action_dim=2, input_dim=4, gamma=0.99, alpha=3e-4, lmda=0.95, 
               clip=0.2, batch_size=batch_size, epochs=4, log_dir='/tmp/ppo')
    
    ppo_agent = ppo_torch.Agent(n_actions=2, input_dims=(4,), gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=batch_size, n_epochs=4)
    
    # Have the agents remember 
    for i in range(num_states):
        skow_agent.remember(states[i], actions[i], log_probs[i], values[i], rewards[i], dones[i])
        ppo_agent.remember(states[i], actions[i], log_probs[i], values[i], rewards[i], dones[i])
    
    # Create the batches 
    state_array, action_array, log_probs_array, value_array, reward_array,\
      done_array, skow_batches = skow_agent.trajectories.generate_batches()
    
    state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, ppo_batches = \
                    ppo_agent.memory.generate_batches()
    
    skow_advantages = skow_agent.calculate_advantages(value_array, reward_array, done_array)
    skow_advantages = torch.tensor(skow_advantages)
    ppo_advantage = ppo_agent.calculate_advantages(vals_arr, reward_arr, dones_arr)
    ppo_advantage = torch.tensor(ppo_advantage)
      # Convert the values to a tensor
    value_array = torch.tensor(value_array)
    vals_arr = torch.tensor(vals_arr)

    # Loop over the training batches
    for batch in skow_batches:
        # Extract the batch
        skow_states = torch.tensor(state_array[batch], dtype=torch.float)
        skow_actions = torch.tensor(action_array[batch])
        skow_old_log_probs = torch.tensor(log_probs_array[batch])

        ppo_states = torch.tensor(state_arr[batch], dtype=torch.float)
        ppo_old_probs = torch.tensor(old_prob_arr[batch])
        ppo_actions = torch.tensor(action_arr[batch])


        # Make fake action probs
        # Step 1: Generate a random tensor of shape (5, 2)
        random_tensor = torch.rand(batch_size, 2)

        # Step 2: Normalize the tensor along the last dimension
        probs = random_tensor / random_tensor.sum(dim=1, keepdim=True)

        # Step 3: Create the Categorical distribution
        categorical_dist = torch.distributions.Categorical(probs=probs)

        skow_new_log_probs = categorical_dist.log_prob(skow_actions)

        ppo_new_probs = categorical_dist.log_prob(ppo_actions)

        # Create critic values 
        # Randomly initialize weights and biases
        weights = torch.randn(batch_size, batch_size, requires_grad=True)  # batch_sizexbatch_size weight matrix
        bias = torch.randn(batch_size, 1, requires_grad=True)     # batch_sizex1 bias vector

        # Input tensor
        input_tensor = torch.randn(batch_size, 1, requires_grad=True)  #batch_sizex1 input vector

        # Perform matrix multiplication and addition
        output_tensor = torch.addmm(bias, weights, input_tensor)
        skow_new_values = torch.squeeze(output_tensor)
        ppo_critic_value = torch.squeeze(output_tensor)

        # Calculate the ratio of log probs
        skow_ratio = skow_new_log_probs.exp()/skow_old_log_probs.exp()
        ppo_prob_ratio = ppo_new_probs.exp() / ppo_old_probs.exp()

        # Calculate the advantage weighted probabilities
        skow_weighted_probs = skow_advantages[batch]*skow_ratio
        skow_weighted_clipped_probs = torch.clamp(skow_ratio, 
                                                1 - skow_agent.clip, 
                                                1 + skow_agent.clip)*skow_advantages[batch]
        

        ppo_weighted_probs = ppo_advantage[batch] * ppo_prob_ratio
        ppo_weighted_clipped_probs = torch.clamp(ppo_prob_ratio, 1-ppo_agent.policy_clip,
                1+ppo_agent.policy_clip)*ppo_advantage[batch]

        # Calculate the policy loss
        policy_loss = -torch.min(skow_weighted_probs, skow_weighted_clipped_probs).mean()
        ppo_actor_loss = -torch.min(ppo_weighted_probs, ppo_weighted_clipped_probs).mean()

        # Calculate the returns from the advantages
        skow_returns = skow_advantages[batch] + value_array[batch]
        ppo_returns = ppo_advantage[batch] + vals_arr[batch]

        # Calculate the value loss
        skow_value_loss = ((skow_returns - skow_new_values)**2).mean()
        ppo_critic_loss = (ppo_returns-ppo_critic_value)**2
        ppo_critic_loss = ppo_critic_loss.mean()

        # Calculate the total loss
        skow_total_loss = policy_loss + 0.5*skow_value_loss
        ppo_total_loss = ppo_actor_loss + 0.5*ppo_critic_loss

        total_loss_match = (skow_total_loss == ppo_total_loss)

        print("Test Loss Vals")
        print(f"    Total Loss Match: {total_loss_match}")


if __name__ == '__main__':

    test_generate_batches()
    test_choose_action()
    test_remember()
    test_advantage()
    test_loss_vals()
