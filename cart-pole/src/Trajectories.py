import numpy as np

class Trajectories:
  def __init__(self, batch_size):
    """
    Initialization function for the Trajectories structure
    """
    self.states = []
    self.actions = []
    self.log_probs = []
    self.values = []
    self.rewards = []
    self.dones = []
    self.batch_size = batch_size

  def generate_batches(self):
    """
    Generates batches of the stored trajectories
    """
    # Extract the number of states stored
    num_states = len(self.states)
    # Determine the start of the batch
    batch_start = np.arange(0, num_states, self.batch_size)
    # Define the indices 
    indices = np.arange(num_states, dtype=np.int64)
    # Randomize the indices 
    np.random.shuffle(indices)
    # Create the batches
    batches = [indices[i:i+self.batch_size] for i in batch_start]
    
    # print(f'states {self.states}')
    # print(f'actions {self.actions}')
    # print(f'log_probs {self.log_probs}')
    # print(f'values {self.values}')
    # print(f'rewards {self.rewards}')
    # print(f'Dones {self.dones}')
    # print(f'Batches {batches}')

    # Return the trajectory info and the batches indeces
    return np.array(self.states),\
            np.array(self.actions),\
            np.array(self.log_probs),\
            np.array(self.values),\
            np.array(self.rewards),\
            np.array(self.dones),\
            batches
    
  def update_trajectory(self, state, action, log_prob, value, reward, done):
    """
    Appends the inputs to the trajectory lists

    :param state: The current state
    :param action: The action taken
    :param log_prob: The log probability of the action
    :param value: The value of the state
    :param reward: The reward for the action
    :param done: Whether the episode is done (1,0)
    """
    self.states.append(state)
    self.actions.append(action)
    self.log_probs.append(log_prob)
    self.values.append(value)
    self.rewards.append(reward)
    self.dones.append(done)

  def clear(self):
    """
    Clears the trajectories
    """
    self.states = []
    self.actions = []
    self.log_probs = []
    self.values = []
    self.rewards = []
    self.dones = []