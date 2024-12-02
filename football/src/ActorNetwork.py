import torch
import torch.nn as nn
import os
from Trajectories import *

class ActorNetwork(nn.Module):
  def __init__(self, input_dim, action_dim, alpha, hidden_dim=256,
               log_dir='/tmp/ppo') :
    """
    Initialization function.  Defines the design of the neural network.
    This network will feature two fully connected layers and a single
    action head.

    :param input_dim: The input dimension (state space)
    :param hidden_dim: The hidden dimension
    :param action_dim: The action dimension (action space)
    """
    super(ActorNetwork, self).__init__()
    # Define the save location for weights
    self.log_dir = log_dir
    self.checkpoint_file = os.path.join(log_dir, 'actor_ppo.pt')
    # Define the network
    self.model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim),
        nn.Softmax(dim=-1)
    )
    # Define the optimizer
    self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, x):
    """
    Defines the forward pass of the neural network

    :param x: The input to the network
    :return: The output of the network
    """
    dist = self.model(x)
    dist = torch.distributions.Categorical(dist)
    return dist

  def save_checkpoint(self, id_str):
    """
    Saves the model weights to a file

    :param id_str: Identifier string added to the file name
    """
    self.checkpoint_file = os.path.join(self.log_dir, f"{id_str}_actor_ppo.pt")
    torch.save(self.state_dict(), self.checkpoint_file)

  def load_checkpoint(self):
    """
    Loads the model weights from a file
    """
    self.load_state_dict(torch.load(self.checkpoint_file))