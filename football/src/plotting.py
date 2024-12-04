from util import *
import numpy as np
import pandas as pd

if __name__=='__main__':
    # Plot the fully trained run-to-score with keeper
    logfile='football/logs/2024-12-01_13-48-56_fully_trained_run_to_score_keeper/academy_run_to_score_with_keeper/training_logs.json'
    save_dir='images/entropy_0.0'
    visualize_logs(logfile, scenario_name='academy_run_to_score_with_keeper', save_dir=save_dir)
    # Empty goal close
    logfile='football/logs/2024-12-03_21-01-41_entropy_0.0/academy_empty_goal_close/training_logs.json'
    visualize_logs(logfile, scenario_name='academy_empty_goal_close', save_dir=save_dir)
    # Empty goal
    logfile='football/logs/2024-12-03_21-09-39_entropy_0.0/academy_empty_goal/training_logs.json'
    visualize_logs(logfile, scenario_name='academy_empty_goal', save_dir=save_dir)
    # Run to score
    logfile='football/logs/2024-12-03_21-09-39_entropy_0.0/academy_run_to_score/training_logs.json'
    visualize_logs(logfile, scenario_name='academy_run_to_score', save_dir=save_dir)
    
    # Plot the partially trained pass_and_shoot_with_keeper
    logfile='football/logs/2024-12-01_13-48-56_fully_trained_run_to_score_keeper/academy_pass_and_shoot_with_keeper/training_logs.json'
    save_dir='images/entropy_0.0'
    visualize_logs(logfile, scenario_name='academy_pass_and_shoot_with_keeper', save_dir=save_dir)

    # Plot the trained empty_goal_close entropy coeff = 0.1
    logfile='football/logs/2024-12-02_07-50-09_entropy_0.1/academy_empty_goal_close/training_logs.json'
    save_dir='images/entropy_0.1'
    visualize_logs(logfile, scenario_name='academy_empty_goal_close', save_dir=save_dir)
    # Plot empty goal entropy coeff 0.1
    logfile='football/logs/2024-12-02_07-50-09_entropy_0.1/academy_empty_goal/training_logs.json'
    visualize_logs(logfile, scenario_name='academy_empty_goal', save_dir=save_dir)

    # Entropy coeff = 0.01
    # empty goal close
    logfile='football/logs/2024-12-02_22-57-38/academy_empty_goal_close/training_logs.json'
    save_dir='images/entropy_0.01'
    visualize_logs(logfile, scenario_name='academy_empty_goal_close', save_dir=save_dir)
    # empty goal
    logfile='football/logs/2024-12-02_22-57-38/academy_empty_goal/training_logs.json'
    visualize_logs(logfile, scenario_name='academy_empty_goal', save_dir=save_dir)
    # run to score
    logfile='football/logs/2024-12-02_22-57-38/academy_run_to_score/training_logs.json'
    visualize_logs(logfile, scenario_name='academy_empty_goal', save_dir=save_dir)
    # Run to score with keeper
    logfile='football/logs/2024-12-02_22-57-38/academy_run_to_score_with_keeper/training_logs.json'
    visualize_logs(logfile, scenario_name='academy_run_to_score_with_keeper', save_dir=save_dir)
