from ast import keyword
from Agent import *
import gfootball.env as football_env
from gfootball.env import football_action_set
import os
import json
import torch
import pandas as pd
from datetime import datetime
from collections import deque
import numpy as np
from util import *


def train_curriculum(academy_scenarios, policy_path='', critic_path=''):
    """
    Trains a curriculum of different environments.

    :param academy_scenarios: The list of of scenario dictionaries
    :param policy_path: Path to initial actor model weights
    :param critic_path: Path to the initial critic model weights
    """
    # Create the curiculum dir
    session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    curriculum_dir = f'/home/bubbles/foundations_rl/ppo/football/logs/{session_timestamp}'
    os.makedirs(curriculum_dir, exist_ok=True)

    # Loop over the scenarios
    prev_scenario_dir = ''
    for scenario in academy_scenarios:
        # Extract the relevant information from the scenario
        scenario_name = scenario["scenario"]
        num_agents = scenario["num_agents"]
        reward_threshold = scenario["reward_threshold"]

        # Create the scenario directory
        scenario_dir = os.path.join(curriculum_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        print(f"Starting to train {scenario_name}")

        # Train the scenario
        is_trained = train(
                scenario_name,
                num_agents,
                reward_threshold,
                policy_path=policy_path,
                critic_path=critic_path,
                log_dir=scenario_dir,
                games=int(35e4),
                num_steps_per_batch=2048,
                batch_size=64,
                epochs=5
        )

        if not is_trained:
            print(f"{scenario_name} failed to train successfully :(")
            # break
        else:
            print(f"Actor Weights: {os.path.join(scenario_dir, 'actor_ppo.pt')}")
            print(f"Critic Weights: {os.path.join(scenario_dir, 'critic_ppo.pt')}")
            prev_scenario_dir = scenario_dir
            policy_path = os.path.join(prev_scenario_dir, '_actor_ppo.pt')
            critic_path = os.path.join(prev_scenario_dir, '_critic_ppo.pt')



def train(env_name, num_agents, reward_threshold, policy_path, critic_path,
                    log_dir, games=1000, num_steps_per_batch=2048, batch_size=64,
                    epochs=4):
    """
    Trains an agent in a specific environment until either the reward threshold is
    met over the average of 100 episodes or the max number of games is reached.
    A log file will be updated each episode and training update.    The model
    weights will be saved off every time the average return suprasses the previous
    maximum.

    :param env_name: The name of the environment
    :param num_agents: The number of agents in the environment
    :param reward_threshold: The reward threshold for the environment
    :param policy_path: The path to the policy weights
    :param critic_path: The path to the critic weights
    :param log_dir: The directory to save logs
    :param games: The max number of games to train
    :param num_steps_per_batch: The number of steps per batch
    :param batch_size: The batch size
    :param epochs: The number of epochs
    :return True: If the training met the threshold
    :return False: If the training reached max games before the threshold
    """
    # Create the environment and extract the dimensions
    is_rendering = False
    env = football_env.create_environment(
            env_name=env_name,
            representation='simple115',
            number_of_left_players_agent_controls=num_agents,
            rewards="scoring,checkpoints",    # Add checkpoint rewards
            render=is_rendering
    )
    action_dim = len(football_action_set.action_set_dict['default'])

    log_data = []

    # Create log dirs
    # Define a checkpoint file
    log_file = os.path.join(log_dir, "training_logs.json")
    os.makedirs(log_dir, exist_ok=True)
    render_dir = os.path.join(log_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    # Create a list of Agents
    agent = Agent(
            action_dim,
            input_dim=max(env.observation_space.shape),
            gamma=0.99,
            alpha=0.0003,
            lmda=0.95,
            clip=0.2,
            batch_size=batch_size,
            epochs=epochs,
            policy_path=policy_path,
            critic_path=critic_path,
            log_dir=log_dir
    )

    best_return = env.reward_range[0]

    # Define arrays to store useful info and counts
    total_rewards = []
    num_updates = 0
    average_return = 0
    steps = 0

    # Loop for the number of games
    for i in range(games):
        # Reset the environment
        observation = env.reset()

        done = False
        total_reward = 0
        policy_losses = []
        value_losses = []
        entropies = []
        episode_frames = []  # List to store frames for the episode
        # Loop until the episode is done
        while not done:
            # Fix the dimension of obs
            if num_agents == 1:
                observation = [observation]

            # Select an action for each agent
            action_list = []
            log_probs_list = []
            value_list = []
            reward_list = []
            for j in range(num_agents):
                # Select an action using the policy
                action, log_probs, value = agent.choose_action(observation[j])
                action_list.append(action)
                log_probs_list.append(log_probs)
            value_list.append(value)

            # Take the action
            next_obs, reward, done, info = env.step(action_list)
            # Check if single agent and add dim to reward
            if num_agents == 1:
                reward = [reward]

            # Increment the step
            steps += 1
            # Update scores
            total_reward += sum(reward)/num_agents
            # Update the5ectories
            for k in range(num_agents):
                agent.remember(observation[k], action, log_probs, value, reward[k], done)
        
            # Capture the frame for rendering
            if is_rendering:
                frame = env.render(mode='rgb_array')
                episode_frames.append(frame)

            # Check if we have met the update number of episodes
            if steps % num_steps_per_batch == 0:
                # Have the agent learn
                policy_loss = 0
                value_loss = 0
                policy_loss, value_loss, entropy = agent.learn()
                # Append the agent average losses
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)
                num_updates += 1
            elif is_rendering and steps % num_steps_per_batch == 1:    # Save a rendering after training
                  # Save the episode frames as a video
                video_path = os.path.join(render_dir, f"episode_{i}.mp4")
                save_video(episode_frames, video_path)

            # Update the observations
            observation = next_obs

        # Update the score history
        total_rewards.append(total_reward)
        # Update the average score
        average_return = np.mean(total_rewards[-100:])
        # Update the best score
        if average_return > best_return:
            best_return = average_return
            agent.save_weights(f"episode_{i}")

        # Calculate the average losses
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy = np.mean(entropies)

        # Print a log
        print('episode', i, 'score %.1f' % total_reward, 'avg score %.1f' % average_return,
                'avg_policy_loss', avg_policy_loss, 'avg_value_loss', avg_value_loss,
                'avg_entropy', avg_entropy, 'time_steps', steps, 'learning_steps', num_updates)

        log_entry = {
            'episode': i,
            'return': total_reward,
            'avg_return': average_return,
            'avg_policy_loss': avg_policy_loss,
            'avg_value_loss': avg_value_loss,
            'avg_entropy': avg_entropy,
            'time_steps': steps,
            'learning_steps': num_updates
        }
        log_data.append(log_entry)

        with open(log_file, "w") as f:
            f.write(json.dumps(log_data))

        # Check if the average score is better than the threshold
        if average_return > reward_threshold and i > 20:
            agent.save_weights("")
            print(f"Training {env_name} complete! Threshold {reward_threshold} met.")
            return True

    print(f"Reached {games} games and failed to meet Threshold {reward_threshold}")
    return False

if __name__ == "__main__":

    # Define the curriculum
    academy_scenarios = [
        {"scenario": "academy_empty_goal_close", "num_agents": 1, "reward_threshold": 1.8}, # For some reason when you score the reward is 2
        {"scenario": "academy_empty_goal", "num_agents": 1, "reward_threshold": 1.8},
        {"scenario": "academy_run_to_score", "num_agents": 1, "reward_threshold": 1.8},
        {"scenario": "academy_run_to_score_with_keeper", "num_agents": 1, "reward_threshold": 1.4},
        {"scenario": "academy_pass_and_shoot_with_keeper", "num_agents": 2, "reward_threshold": 1.4},
        {"scenario": "academy_run_pass_and_shoot_with_keeper", "num_agents": 2, "reward_threshold": 1.4},
        {"scenario": "academy_3_vs_1_with_keeper", "num_agents": 3, "reward_threshold": 1.4},
        {"scenario": "academy_corner", "num_agents": 11, "reward_threshold": 1.4},
        {"scenario": "academy_counterattack_easy", "num_agents": 4, "reward_threshold": 1.4},
        {"scenario": "academy_counterattack_hard", "num_agents": 4, "reward_threshold": 1.4},
        {"scenario": "academy_single_goal_versus_lazy", "num_agents": 11, "reward_threshold": 1.4},
        {"scenario": "11_vs_11_easy_stochastic", "num_agents": 11, "reward_threshold": 1.4},
        {"scenario": "11_vs_11_stochastic", "num_agents": 11, "reward_threshold": 1.4},
        {"scenario": "11_vs_11_hard_stochastic", "num_agents": 11, "reward_threshold": 1.4}
    ]

    #policy_path = '/home/bubbles/foundations_rl/ppo/football/logs/2024-12-01_13-48-56/academy_run_to_score_with_keeper/_actor_ppo.pt'
    #critic_path = '/home/bubbles/foundations_rl/ppo/football/logs/2024-12-01_13-48-56/academy_run_to_score_with_keeper/_critic_ppo.pt'
    # policy_path='/home/bubbles/foundations_rl/ppo/football/logs/2024-12-01_20-08-22/academy_empty_goal/_actor_ppo.pt'
    # critic_path='/home/bubbles/foundations_rl/ppo/football/logs/2024-12-01_20-08-22/academy_empty_goal/_critic_ppo.pt'
    policy_path='/home/bubbles/foundations_rl/ppo/football/logs/2024-12-03_21-01-41/academy_empty_goal_close/_actor_ppo.pt'
    critic_path='/home/bubbles/foundations_rl/ppo/football/logs/2024-12-03_21-01-41/academy_empty_goal_close/_critic_ppo.pt'
    train_curriculum(academy_scenarios, policy_path, critic_path)
