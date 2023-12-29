
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense import DenseGCNConv

class GCNFacilityEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCNFacilityEmbedding, self).__init__()
        # self.conv1 = GCNConv(input_dim, hidden_dim)
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.conv3 = GCNConv(hidden_dim, output_dim)
        self.conv1 = DenseGCNConv(input_dim, hidden_dim)
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        self.conv3 = DenseGCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    # def forward(self, fac_matrix, edge_index, edge_weight):
    def forward(self, x, adj_mat):
        x = F.relu(self.conv1(x, adj_mat))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, adj_mat))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, adj_mat)

class ConvLocationEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, kernel_size):
        super(ConvLocationEmbedding, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(input_dim, embed_dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=padding)
        self.conv3 = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=padding)

    def forward(self, locations):
        return F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(locations)))))) # (batch_size, input_dim, n) -> (batch_size, embed_dim, n)
 
# an attention block consiting of three following steps:
# location embeddings: L ∈ R d_k × n
# Extended location embeddings by appending GRU output to L: L˜ ∈ R 2d_k × n
# (1) a⊤ = softmax(v⊤_a f(W_a L˜))
# (2) c = La 
# (3) o⊤ = softmax(v⊤_o f(W_o [L˜; c]))
class AttentionBlock(nn.Module):
    def __init__(self, attn_dim, embed_dim):
        super(AttentionBlock, self).__init__()

        self.v_a = nn.Parameter(torch.Tensor(attn_dim))
        self.W_a = nn.Parameter(torch.Tensor(attn_dim, 2*embed_dim))

        self.v_o = nn.Parameter(torch.Tensor(attn_dim))
        self.W_o = nn.Parameter(torch.Tensor(attn_dim, 2*embed_dim))

        self.init_parameters()
    
    def init_parameters(self):
        # nn.init.xavier_uniform_(self.v_a)
        nn.init.normal_(self.v_a)
        nn.init.xavier_uniform_(self.W_a)
        # nn.init.xavier_uniform_(self.v_o)
        nn.init.normal_(self.v_o)
        nn.init.xavier_uniform_(self.W_o)

    def forward(self, embeddings, gru_output, action_mask):
        # embeddings: (batch_size, n, embed_dim)
        # gru_output: (batch_size, 1, embed_dim)

        # (1) a⊤ = softmax(v⊤_a f(W_a L˜))
        n = embeddings.size(1)

        a = F.softmax(
            torch.matmul(
                F.relu(
                   torch.matmul(
                        torch.cat((embeddings, gru_output.repeat(1,n,1)), dim=2),
                        self.W_a.T
                        )
                    ), 
                self.v_a
                ),
            dim=1) # -> (batch_size, n)

        # (2) c = La
        # c = torch.matmul(a, embeddings)
        c = torch.bmm(a.unsqueeze(1), embeddings) # -> (batch_size, 1, embed_dim)

        # (3) o⊤ = softmax(v⊤_o f(W_o [L˜; c]))
        o_score = torch.matmul(
            F.relu(
                torch.matmul(
                    torch.cat((embeddings, c.repeat(1,n,1)), dim=2),
                    self.W_o.T)
                ),
            self.v_o
            )
        o = F.softmax(
            o_score + action_mask,
            dim=1
            ) # -> (batch_size, n)
        
        return o
        

# pointer block
class PointerBlock(nn.Module):
    def __init__(self, attn_dim, embed_dim):
        super(PointerBlock, self).__init__()
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.attention = AttentionBlock(attn_dim, embed_dim)

    def forward(self, last_selected_emb, hidden_state, embeddings, action_mask): #, fac_edge_index, fac_edge_weight, L_edge_index, L_edge_weight):
        gru_output, gru_hidden_state = self.gru(
            last_selected_emb, hidden_state
            )
        out_probs = self.attention(embeddings, gru_output, action_mask)
        return out_probs, gru_output, gru_hidden_state


class DoublePointerNetwork(nn.Module):
    def __init__(self, num_instances, loc_input_dim, fac_input_dim, embed_dim, attn_dim, dropout, device):
        super(DoublePointerNetwork, self).__init__()
        self.dropout = dropout
        self.num_instances = num_instances
        self.device = device
        self.embed_dim = embed_dim

        self.loc_embedding_layer = ConvLocationEmbedding(loc_input_dim, embed_dim, kernel_size=7).to(device)
        self.fac_embedding_layer = GCNFacilityEmbedding(fac_input_dim, embed_dim, embed_dim, dropout).to(device)

        self.U_ptr= PointerBlock(attn_dim, embed_dim).to(device)
        self.L_ptr= PointerBlock(attn_dim, embed_dim).to(device)


    def select_action(self, embeddings, probs):
        # Sample an action from the action probability distribution
        # embeddings: (batch_size, n, embed_dim)
        # probs: (batch_size, n)
        # return: (batch_size), (batch_size, embed_dim)
        batch_size = embeddings.size(0)
        with torch.no_grad():
            m = Categorical(probs)
            action = m.sample() # -> (batch_size)
            # log_pi_a_t, entropy = m.log_prob(action), m.entropy().mean() # -> (batch_size), scalar
        
        action_emb = embeddings[torch.arange(batch_size), action, :]
        return action, action_emb #, log_pi_a_t, entropy

    def forward(self, fac_matrix, locations):
        
        batch_size = fac_matrix.size(0)
        n = fac_matrix.size(1)
        embed_dim = self.embed_dim
        locations = locations.to(self.device)
        fac_matrix = fac_matrix.to(self.device)

        fac_one_hot = torch.eye(n).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device) # (batch_size, n, n)

        # encode locations
        loc_embeddings = self.loc_embedding_layer(locations.permute(0,2,1)).permute(0,2,1) # (batch_size, n, 2) -> (batch_size, n, embed_dim)

        # encode facilities
        fac_embeddings = self.fac_embedding_layer(fac_one_hot, fac_matrix) # (batch_size, n, inp) -> (batch_size, n, embed_dim)

        # initialize the starting `last selected embedding`
        U_last_selected_emb = torch.zeros(batch_size, 1, embed_dim).to(self.device)
        L_last_selected_emb = torch.zeros(batch_size, 1, embed_dim).to(self.device)

        # initialize the starting hidden state
        U_hidden = torch.zeros(1, batch_size, embed_dim).to(self.device)
        L_hidden = torch.zeros(1, batch_size, embed_dim).to(self.device)

        # initialize probs
        U_probs = torch.zeros(n, batch_size, n).to(self.device)
        L_probs = torch.zeros(n, batch_size, n).to(self.device)

        # initialize selected indicies
        U_selected_locs = torch.zeros(n, batch_size).to(self.device)
        L_selected_facs = torch.zeros(n, batch_size).to(self.device)

        # action masking to prevent selecting previously selected actions
        location_mask = torch.zeros(batch_size, n).to(self.device)
        facility_mask = torch.zeros(batch_size, n).to(self.device)


        # loop over all instances (Main Double Pointer Network loop)
        for i in range(self.num_instances):
            # U_ptr
            # Note that the first time-step is a dummy step,
            # where the last selected embedding is a zero vector
            # Also the input to "U" is selected by "L" and vice versa

            # pass action mask into pointer block
            U_out_probs, _, U_hidden = self.U_ptr(
                L_last_selected_emb, U_hidden, loc_embeddings, location_mask
                )
            U_probs[i] = U_out_probs
            U_selected_idx, U_selected_emb = self.select_action(loc_embeddings, U_out_probs) # selects location
            U_selected_locs[i] = U_selected_idx

            U_last_selected_emb[:,0,:] = U_selected_emb
            # update mask
            location_mask[torch.arange(batch_size), U_selected_idx] = -float('inf')


            # L_ptr
            # pass action mask into pointer block
            L_out_probs, _, L_hidden = self.L_ptr(
                U_last_selected_emb, L_hidden, fac_embeddings, facility_mask
                )
            L_probs[i] = L_out_probs
            L_selected_idx, L_selected_emb = self.select_action(fac_embeddings, L_out_probs) # selects assigned facility
            L_selected_facs[i] = L_selected_idx

            L_last_selected_emb[:,0,:] = L_selected_emb
            # update mask
            facility_mask[torch.arange(batch_size), L_selected_idx] = -float('inf')

        return U_probs, L_probs, U_selected_locs, L_selected_facs


# critic network
class Critic(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.linear3 = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        # x: (batch_size, 2(n^2) + 2n))
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.linear3(x)