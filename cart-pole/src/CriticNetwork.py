import torch
import torch.nn as nn
import os

class CriticNetwork(nn.Module):
  def __init__(self, input_dim, alpha, hidden_dim=256, 
               log_dir='/tmp/ppo'):
    """
    The initialization function defines the design of the critic network.
    This network will feature two fully connected layers and a single value 
    head.

    :param input_dim: The input dimension (state space)
    :param hidden_dim: The hidden dimension
    """
    super(CriticNetwork, self).__init__()

    # Define a checkpoint file
    self.checkpoint_file = os.path.join(log_dir, 'critic_ppo.pt')
    # Define the network
    self.model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )
    # Define the optimizer
    self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
    # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    # self.to(self.device

  def forward(self, x):
    """
    Defines the forward pass of the network

    :param x: The input to the network
    :return: The output of the network
    """
    value = self.model(x)
    return value
  
  def save_checkpoint(self):
    """
    Saves the model weights to a file
    """
    torch.save(self.state_dict(), self.checkpoint_file)

  def load_checkpoint(self):
    """
    Loads the model weights from a file
    """
    self.load_state_dict(torch.load(self.checkpoint_file))
