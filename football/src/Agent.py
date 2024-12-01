import torch
import numpy as np

class Agent:
  def __init__(self, action_dim, input_dim, gamma=0.99, alpha=3e-4, lmda=0.95,
               clip=0.2, batch_size=64, epochs=10, policy_path='',
               critic_path='', log_dir='/tmp/ppo'):
    """
    Initialization function for the Agent class.  Defines useful class variables
    and objects.

    :param action_dim: The number of actions the agent can take
    :param input_dim: The state space dimension
    :param gamma: The discount factor
    :param alpha: The learning rate
    :param lmda: The tradeoff parameter
    :param clip: The clipping parameter
    :param batch_size: The batch size
    :param epochs: The number of epochs
    :param policy_path: The path to the policy weights
    :param critic_path: The path to the critic weights
    :param log_dir: The directory to save logs
    """
    # Load the variables
    self.action_dim = action_dim
    self.input_dim = input_dim
    self.gamma = gamma
    self.alpha = alpha
    self.lmda = lmda
    self.clip = clip
    self.batch_size = batch_size
    self.epochs = epochs

    # Create the actor and critic networks
    self.actor = ActorNetwork(input_dim, action_dim, alpha, log_dir=log_dir)
    self.critic = CriticNetwork(input_dim, alpha, log_dir=log_dir)
    # Create the trajectories
    self.trajectories = Trajectories(batch_size)

    # Check if the weights paths exist
    if os.path.exists(policy_path):
      self.actor.load_state_dict(torch.load(policy_path))
      print(f"Loaded policy weights from {policy_path}")
    else:
      print(f"No policy weights found at {policy_path}")
    if os.path.exists(critic_path):
      self.critic.load_state_dict(torch.load(critic_path))
      print(f"Loaded critic weights from {critic_path}")
    else:
      print(f"No critic weights found at {critic_path}")

  def remember(self, state, action, log_prob, value, reward, done):
    """
    Stores data to the trajectories

    :param state: The current state(s)
    :param action: The action(s) taken
    :param log_prob: The log probability of the action(s)
    :param value: The value(s) of the state(s)
    :param reward: The reward(s) for the action(s)
    :param done: Whether the episode is terminal (0,1)
    """
    self.trajectories.update_trajectory(state, action, log_prob, value, reward, done)

  def save_weights(self):
    """
    Saves the model weights for the actor and the critic
    """
    print("Saving model weights...")
    self.actor.save_checkpoint()
    self.critic.save_checkpoint()

  def load_weights(self):
    """
    Loads the model weights for the actor and the critic
    """
    print("Loading model weights...")
    self.actor.load_checkpoint()
    self.critic.load_checkpoint()

  def choose_action(self, observation):
    """
    Selects an action based on the output of the actor policy model

    :param observation: The current state to select an action for
    :return action: The action to take
    :return log_probs: The log probability of the action
    :return value: The value estimate from the critic
    """
    # Convert the observation to a tensor
    state = torch.tensor([observation], dtype=torch.float)
    # Obtain the action probabilities
    action_probs = self.actor(state)
    # Sample the action
    action = action_probs.sample()

    # Obtain the value estimate from the critic
    value = self.critic(state)

    # Fix the dimensions of the output
    log_probs = torch.squeeze(action_probs.log_prob(action)).item()
    action = torch.squeeze(action).item()
    value = torch.squeeze(value).item()

    return action, log_probs, value

  def calculate_advantages(self, value_array, reward_array, done_array):
    """
    Calculates the advantages array from the values and rewards in the batch

    :param value_array: The batched values
    :param reward_array: The batched rewards
    :param done_array: The batched dones
    :return advantages: Array of advantage values
    """

    # Rename values
    values = value_array
    # initialize the advantages
    advantages = np.zeros(len(reward_array), dtype=np.float32)

    # Loop through each index
    for t in range(len(reward_array) - 1):
      # Initialize the GAE discount and the advantage
      discount = 1
      advantage_t = 0
      # Loop from current t until the end
      for k in range(t, len(reward_array) - 1):
        # Calculate the advantage
        advantage_t += discount*(reward_array[k] + self.gamma*values[k + 1]*\
                                (1 - int(done_array[k])) - values[k])
        # Update the discount
        discount *= self.gamma*self.lmda

      # Store the advantage
      advantages[t] = advantage_t

    return advantages


  def learn(self):
    """
    Instantiation of the proximal policy optimization algorithm
    """
    # Define arrays to store the loss values
    policy_losses = []
    value_losses = []
    # Loop over every epoch
    for _ in range(self.epochs):
      state_array, action_array, log_probs_array, value_array, reward_array,\
      done_array, batches = self.trajectories.generate_batches()
      ### Calculate Advantages ###
      advantages = self.calculate_advantages(value_array, reward_array, done_array)
      # Convert the advantages to a tensor
      advantages = torch.tensor(advantages)
      # Convert the values to a tensor
      values = torch.tensor(value_array)
      ### Calculate Advantages ###

      # Loop over the training batches
      for batch in batches:
        # Extract the batch
        states = torch.tensor(state_array[batch], dtype=torch.float)
        actions = torch.tensor(action_array[batch])
        old_log_probs = torch.tensor(log_probs_array[batch])

        # Extract the new action and log probs from the actor
        new_action_probs = self.actor(states)
        new_log_probs = new_action_probs.log_prob(actions)

        # Extract the new values from the critic
        new_values = self.critic(states)
        new_values = torch.squeeze(new_values)

        # Calculate the ratio of log probs
        ratio = new_log_probs.exp()/old_log_probs.exp()

        # Calculate the advantage weighted probabilities
        weighted_probs = advantages[batch]*ratio
        weighted_clipped_probs = torch.clamp(ratio,
                                             1 - self.clip,
                                             1 + self.clip)*advantages[batch]
        # Calculate the policy loss
        policy_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
        # print(policy_loss)
        policy_losses.append(policy_loss.item())

        # Calculate the returns from the advantages
        returns = advantages[batch] + values[batch]

        # Calculate the value loss
        value_loss = ((returns - new_values)**2).mean()
        value_losses.append(value_loss.item())

        # Calculate the total loss
        total_loss = policy_loss + 0.5*value_loss

        # Zero grad the optimizers
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        # Calculate the gradients
        total_loss.backward()

        # Update the weights
        self.actor.optimizer.step()
        self.critic.optimizer.step()

    # Clear the trajectories
    self.trajectories.clear()

    return np.mean(policy_losses), np.mean(value_losses)