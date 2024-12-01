import gym
import numpy as np
from Agent import Agent
import os
from datetime import datetime
import json
from util import *

if __name__ == '__main__':
    # Create the environment and extract the dimensions
    env = gym.make("CartPole-v1")
    num_steps_per_batch = 20
    batch_size = 5
    epochs = 4
    log_data = []

    # Create log dirs
    # Define a checkpoint file
    session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'/Users/andrewskow/Documents/JHU_EP/foundations_rl/project/skow_ppo/logs/{session_timestamp}'
    log_file = os.path.join(log_dir, "training_logs.json")
    os.makedirs(log_dir, exist_ok=True)

    agent = Agent(
        action_dim=env.action_space.n,
        input_dim=env.observation_space.shape[0],
        gamma=0.99,
        alpha=0.0003,
        lmda=0.95,
        clip=0.2,
        batch_size=batch_size,
        epochs=epochs,
        log_dir=log_dir
    )

    games = 300

    best_score = env.reward_range[0]

    # Define arrays to store useful info and counts 
    scores = []
    num_updates = 0
    average_score = 0
    steps = 0

    # Loop for the number of games
    for i in range(games):
        # Reset the environment
        observation = env.reset()
        if gym.__version__>'0.26.0':
            observation = observation[0]

        done = False
        score = 0
        policy_losses = []
        value_losses = []
        # Loop until the episode is done
        while not done:
            # Select an action using the policy
            action, log_probs, value = agent.choose_action(observation)
            # Take the action
            if gym.__version__>'0.26.0':
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                next_obs, reward, done, info = env.step(action)
            # Increment the step
            steps += 1
            # Update scores
            score += reward
            # Update the5ectories
            agent.remember(observation, action, log_probs, value, reward, done)

            # Check if we have met the update number of episodes
            if steps % num_steps_per_batch == 0:
                # Have the agent learn
                policy_loss, value_loss = agent.learn()
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                num_updates += 1
            
            # Update the observations
            observation = next_obs
        
        # Update the score history
        scores.append(score)
        # Update the average score
        average_score = np.mean(scores[-100:])
        # Update the best score
        if average_score > best_score:
            best_score = average_score
            agent.save_weights()
        
        # Calculate the average losses
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        
        # Print a log 
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % average_score, 
                'avg_policy_loss', avg_policy_loss, 'avg_value_loss', avg_value_loss,
                'time_steps', steps, 'learning_steps', num_updates)
        
        log_entry = {
            'episode': i,
            'score': score,
            'avg_score': average_score,
            'avg_policy_loss': avg_policy_loss,
            'avg_value_loss': avg_value_loss,
            'time_steps': steps,
            'learning_steps': num_updates
        }
        log_data.append(log_entry)

        with open(log_file, "w") as f:
            f.write(json.dumps(log_data))