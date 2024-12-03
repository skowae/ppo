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

def create_renderings(num_vids, scenario_name, num_agents, policy_path, critic_path, save_dir, file_id):
    """
    Creates a number of renderings of the agent playing in the scenario with the given policy 
    and critic models.  The function will create a football environment and and agent to create 
    the renderings
    
    :param num_vids: The number of video renderings to make
    :param scenario_name: The environment scenario name
    :param num_agents: The number of agents in the environment
    :param policy_path: The file path to the policy model weights
    :param critic_path: The file path to the critic model weights
    :param save_dir: The directory to save the renderings
    """
    # Create the environment and extract the dimensions
    env = football_env.create_environment(
            env_name=scenario_name,
            representation='simple115',
            number_of_left_players_agent_controls=num_agents,
            rewards="scoring,checkpoints",    # Add checkpoint rewards
            render=True
    )
    action_dim = len(football_action_set.action_set_dict['default'])

    log_data = []

    # Create log dirs
    # Define a checkpoint file
    log_file = os.path.join(save_dir, f"{file_id}_rendering_logs.json")
    os.makedirs(save_dir, exist_ok=True)

    # Create the Agents
    agent = Agent(
            action_dim,
            input_dim=max(env.observation_space.shape),
            policy_path=policy_path,
            critic_path=critic_path,
    )

    best_return = env.reward_range[0]

    # Define arrays to store useful info and counts
    total_rewards = []
    num_updates = 0
    average_return = 0
    steps = 0

    # Loop for the number of games
    for i in range(num_vids):
        # Reset the environment
        observation = env.reset()

        done = False
        total_reward = 0
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
            frame = env.render(mode='rgb_array')
            episode_frames.append(frame)

            # Save the episode frames as a video
            video_path = os.path.join(save_dir, f"{file_id}_vid_{i}.mp4")
            save_video(episode_frames, video_path)

            # Update the observations
            observation = next_obs

        # Update the score history
        total_rewards.append(total_reward)
        # Update the average score
        average_return = np.mean(total_rewards[-100:])

        # Print a log
        print('episode', i, 'score %.1f' % total_reward, 'avg score %.1f' % average_return,
                'time_steps', steps, 'learning_steps', num_updates)

        log_entry = {
            'episode': i,
            'return': total_reward,
            'avg_return': average_return,
            'time_steps': steps,
            'learning_steps': num_updates
        }
        log_data.append(log_entry)

        with open(log_file, "w") as f:
            f.write(json.dumps(log_data))
            
    print(f"Renderings saved to {save_dir}")

if __name__ == "__main__":

    # Define the input variables
    num_vids = 10
    scenario_name = "academy_empty_goal_close"
    num_agents = 1
    policy_path = "/home/bubbles/foundations_rl/ppo/football/logs/2024-12-02_18-45-47/academy_empty_goal_close/_actor_ppo.pt"
    critic_path = "/home/bubbles/foundations_rl/ppo/football/logs/2024-12-02_18-45-47/academy_empty_goal_close/_critic_ppo.pt"
    save_dir = "/home/bubbles/foundations_rl/ppo/football/logs/2024-12-02_18-45-47/academy_empty_goal_close/renders"
    file_id = "final_wieghts"
    # Create the renderings
    create_renderings(num_vids, scenario_name, num_agents, policy_path, critic_path, save_dir, file_id)