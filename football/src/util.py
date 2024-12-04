import json
import matplotlib.pyplot as plt
import os

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

import cv2
import numpy as np

def save_video(frames, path, fps=30):
    """
    Saves a fide from a list of frames for rendering the environment

    :param frames: The list of vide frames from the env
    :param path: The path to save the renderings
    :param fps: The frames per second of the video
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    video.release()


def smooth_plot(log_data, metric_key, window_size=10, scenario_name="", ylabel="Value", save_dir='./'):
  """
  Generate a smoothed plot where each y value is an average over a rolling window of epochs.
  The x-axis reflects the true epoch numbers.

  :param log_data: List of log entries
  :param metric_key: The key in the log entries to plot (e.g., "average_reward", "policy_loss")
  :param window_size: Number of epochs to average over for smoothing
  :param scenario_name: Title of the plot for context
  :param ylabel: Y-axis label for the plot
  """
  # Check if metric_key exists in the data
  if any(metric_key in entry for entry in log_data):
      file_name = os.path.join(save_dir, f'{scenario_name}_{metric_key}.png')

      # Extract data, ignoring entries where the metric is NaN
      epochs = [entry["episode"] for entry in log_data if metric_key in entry and not np.isnan(entry[metric_key])]
      metrics = [entry[metric_key] for entry in log_data if metric_key in entry and not np.isnan(entry[metric_key])]

      # Check if there is enough data to create a plot
      if len(metrics) > 0:
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
          plt.savefig(file_name)
          plt.close()
      else:
          print(f"No valid data to plot for metric_key: {metric_key}")
  else:
      print(f"metric_key '{metric_key}' not found in the log data.")

def visualize_logs(log_file, scenario_name, save_dir):
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
  smooth_plot(log_data, "return", window_size=10, scenario_name=scenario_name, ylabel="Return", save_dir=save_dir)
  smooth_plot(log_data, "avg_return", window_size=10, scenario_name=scenario_name, ylabel="Average Return", save_dir=save_dir)
  smooth_plot(log_data, "avg_policy_loss", window_size=10, scenario_name=scenario_name, ylabel="Average Policy Loss", save_dir=save_dir)
  smooth_plot(log_data, "avg_value_loss", window_size=10, scenario_name=scenario_name, ylabel="Average Value Loss", save_dir=save_dir)
  smooth_plot(log_data, "entropy", window_size=10, scenario_name=scenario_name, ylabel="Average Entropy", save_dir=save_dir)
