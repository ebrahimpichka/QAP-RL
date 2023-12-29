import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


from models import DoublePointerNetwork, Critic
from utils import generate_batch_qap_problem_instance


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
    loc_input_dim = 2 
    fac_input_dim = 20 # one-hot encoding of the facilities as initial input feature
    attn_dim = 256
    embed_dim = 512
    dropout = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 50
    epochs = 20
    optimizer = 'adam' # 'sgd'



class Agent(nn.Module):
    """The Agent class.

    Attributes:
        config: Config object.
    """

    def __init__(self, config):
        """Initializes the trainer.

        Args:
            config: nonfig object.
        """
        super().__init__()

        self.config = config

        self.model = DoublePointerNetwork(
            num_instances=config.num_instances, 
            loc_input_dim=config.loc_input_dim, 
            fac_input_dim=config.fac_input_dim, 
            embed_dim=config.embed_dim, 
            attn_dim=config.attn_dim, 
            dropout=config.dropout, 
            device=config.device
            )
        
        self.critic = Critic(
            2*(config.num_instances**2)+2*config.num_instances, # 2*(n^2) + 2*n
            config.embed_dim, 
            config.dropout
            )
        
    def get_optimizer(self, opt_type='adam'):
        if opt_type=='adam':
            return torch.optim.Adam(self.parameters(), lr=0.0001)
    
    
    def train(self):
        """Trains the model.
        """

        optimizer = self.get_optimizer()

        for epoch in range(self.config.epochs):
            locations, _, distance_matrix, fac_matrix =\
                generate_batch_qap_problem_instance(self.config.batch_size, self.config.num_instances)
            
            batch_size = fac_matrix.size(0)
            num_instances = fac_matrix.size(1)

            
            # perform batch rollout
            U_probs, L_probs, U_selected_locs, L_selected_facs =\
                self.model(
                    fac_matrix=fac_matrix.float(),
                    locations=locations.float()
                    )


            # calculate the reward trajectory
            reward_trajectory =\
                  calc_trajectory_reward(fac_matrix, distance_matrix, U_selected_locs, L_selected_facs)

            # create value states
            states = torch.zeros(batch_size, num_instances, 2*(num_instances**2)+2*num_instances)
            for step in range(0, 2*num_instances, 2):
                states[] = ... # TODO
            
            values = self.critic(states) 

            # TODO
            # calc returns and advantages based on V(s_t) and V(s_t+1) and r_t
            advantage = 

            # calc loss 
            # update params
            U_dist = Categorical(probs=U_probs)
            U_actor_loss = -U_dist.log_prob(U_selected_locs) * advantage # TODO

            L_dist = Categorical(probs=L_probs)
            L_actor_loss = -L_dist.log_prob(L_selected_facs) * advantage # TODO


            U_advantage_loss = ... # TODO
            L_advantage_loss = ... # TODO

            U_entropy_loss = ... # TODO
            L_entropy_loss = ... # TODO

            # TODO
            U_loss = U_actor_loss + self.config.alpha* U_advantage_loss + self.config.beta*U_entropy_loss
            L_loss = L_actor_loss + self.config.alpha* L_advantage_loss + self.config.beta*L_entropy_loss

            loss = U_loss + L_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




