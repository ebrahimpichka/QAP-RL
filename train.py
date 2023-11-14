import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DoublePointerNetwork, Critic


def calc_trajectory_reward(fac_matrix, loc_matrix, loc_trajectory, fac_trajectory):
    """Calculates the reward of a given trajectory.

    Args:
		fac_matrix: The facilities flow matrix.
		loc_matrix: The location distance matrix.
		loc_trajectory: The selected locations trajectory. (num_instances, batch_size)
		fac_trajectory: The selected facilities trajectory.(num_instances, batch_size)

    Returns:
		The reward trajectory.
    """
    batch_size = fac_matrix.size(0)
    num_instances = fac_matrix.size(1)
    reward_trajectory = torch.zeros(2* num_instances, batch_size)
    
    for timestep in range(0, 2*num_instances, 2):
        reward_trajectory[timestep] = torch.zeros(batch_size)

        selected_locs = loc_trajectory[timestep//2]
        selected_facs = fac_trajectory[timestep//2]

        up2here_locs = loc_trajectory[:timestep//2 - 1]
        up2here_facs = fac_trajectory[:timestep//2 - 1]

        batch_rewards = torch.zeros(batch_size)
        for batch_idx in range(batch_size):
            single_up2here_locs = up2here_locs[:, batch_idx]
            single_up2here_facs = up2here_facs[:, batch_idx]

            selected_loc = selected_locs[batch_idx]
            selected_fac = selected_facs[batch_idx]
            
            
            reward = 0

            for loc, fac in zip(single_up2here_locs, single_up2here_facs):
                dist = loc_matrix[batch_idx, loc, selected_loc]
                flow = fac_matrix[batch_idx, fac, selected_fac]
                reward += 2*(dist * flow)
            batch_rewards[batch_idx] = reward
 
        reward_trajectory[timestep+1] = batch_rewards

    return reward_trajectory

class Config():
    """The config class."""
    num_instances = 20 # (n)
    input_dim = 2
    embed_dim = 512
    dropout = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 50
    epochs = 20

class Trainer():
    """The trainer class.

    Attributes:
        config: nonfig object.
    """

    def __init__(self, config):
        """Initializes the trainer.

        Args:
            config: nonfig object.
        """

        self.config = config

        self.model = DoublePointerNetwork(
            config.num_instances, 
            config.input_dim, 
            config.embed_dim, 
            config.dropout, 
            config.device
            )
        
        self.critic = Critic(
            config.num_instances, 
            config.embed_dim, 
            config.dropout, 
            config.device
            )

    def train(self):
        """Trains the model.

        """
        batch_size = fac_matrix.size(0)
        num_instances = fac_matrix.size(1)

        # calculate the reward trajectory
        reward_trajectory = calc_trajectory_reward(fac_matrix, loc_matrix, loc_trajectory, fac_trajectory)

        # calculate the critic trajectory
        critic_trajectory = torch.zeros(2*num_instances, batch_size)
        for timestep in range(0, 2*num_instances, 2):
            critic_trajectory[timestep] = torch.zeros(batch_size)

            selected_locs = loc_trajectory[timestep//2]
            selected_facs = fac_trajectory[timestep//2]

            up2here_locs = loc_trajectory[:timestep//2 - 1]
            up2here_facs = fac_trajectory[:timestep//2 - 1]

            batch_rewards = torch.zeros(batch_size)
            for batch_idx in range(batch_size):
                single_up2here_locs = up2here_locs[:, batch_idx]
                single_up2here_facs = up2here_facs[:, batch_idx]

                selected_loc = selected_locs[batch_idx]
                selected_fac = selected_facs[batch_idx]
                
                
                reward = 0

                for loc, fac in zip(single_up2here_locs, single_up2here_facs):
                    dist =
    


