"""
Revised from CommNetMLP
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from models import MLP
import sys 
sys.path.append("..") 
from action_utils import select_action, translate_action

class GACommNetMLP(nn.Module):
    def __init__(self, args, num_inputs):
        super(GACommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent
        self.qk_hid_size = args.qk_hid_size
        

        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            # support multi action
            self.heads = nn.ModuleList([nn.Linear(args.hid_size*2, o)
                                        for o in args.naction_heads])
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)

        self.encoder = nn.Linear(num_inputs, args.hid_size)

        if args.recurrent:
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        if args.recurrent:
            self.init_hidden(args.batch_size)
            self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)

        else:
            if args.share_weights:
                self.f_module = nn.Linear(args.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                                for _ in range(self.comm_passes)])
        # else:
            # raise RuntimeError("Unsupported RNN type.")

        # Our main function for converting current hidden state to next state
        # self.f = nn.Linear(args.hid_size, args.hid_size)
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                            for _ in range(self.comm_passes)])
        # self.C = nn.Linear(args.hid_size, args.hid_size)

        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        # print(self.C)
        # self.C.weight.data.zero_()
        # Init weights for linear layers
        # self.apply(self.init_weights)

        self.value_head = nn.Linear(2*self.hid_size, 1)

        # hard attention layers to form the graph 
        self.lstm = nn.LSTM(args.hid_size * 2, args.hid_size * 2, bidirectional=True)
        self.linear = nn.Linear(args.hid_size * 4, 2) # *4: after bidirectional output
        # soft attention layers 
        self.wq = nn.Linear(args.hid_size, args.qk_hid_size)
        self.wk = nn.Linear(args.hid_size, args.qk_hid_size)


    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()

        return num_agents_alive, agent_mask

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, extras = x
            x = self.encoder(x)

            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
        else:
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state


    def forward(self, x, info={}):
        # TODO: Update dimensions
        """Forward function for CommNet class, expects state, previous hidden
        and communication tensor.
        B: Batch Size: Normally 1 in case of episode
        N: number of agents
        Arguments:
            x {tensor} -- State of the agents (N x num_inputs)
            prev_hidden_state {tensor} -- Previous hidden state for the networks in
            case of multiple passes (1 x N x hid_size)
            comm_in {tensor} -- Communication tensor for the network. (1 x N x N x hid_size)
        Returns:
            tuple -- Contains
                next_hidden {tensor}: Next hidden state for network
                comm_out {tensor}: Next communication tensor
                action_data: Data needed for taking next action (Discrete values in
                case of discrete, mean and std in case of continuous)
                v: value head
        """

        # if self.args.env_name == 'starcraft':
        #     maxi = x.max(dim=-2)[0]
        #     x = self.state_encoder(x)
        #     x = x.sum(dim=-2)
        #     x = torch.cat([x, maxi], dim=-1)
        #     x = self.tanh(x)

        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        # # Hard Attention - action whether an agent communicates or not
        # if self.args.hard_attn:
        #     comm_action = torch.tensor(info['comm_action'])
        #     comm_action_mask = comm_action.expand(batch_size, n).unsqueeze(-1)
        #     # action 1 is talk, 0 is silent i.e. act as dead for comm purposes. ???
        #     agent_mask *= comm_action_mask.double()            

        # Initialize communication tensor
        c = torch.zeros(batch_size * n, self.hid_size).to(x.device)
        comm_density1 = 0
        comm_density2 = 0

        for i in range(self.comm_passes):
            if self.args.recurrent:
                inp = x + c
                inp = inp.view(batch_size * n, self.hid_size)
                output = self.f_module(inp, (hidden_state, cell_state))
                hidden_state = output[0]
                cell_state = output[1]
            else: # MLP|RNN
                # Get next hidden state from f node
                # and Add skip connection from start and sum them
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)

            if not self.args.comm_mask_zero:
                if not self.args.comm_action_one:
                    # TO DO: should do batch-wise
                    h0 = torch.zeros(2, batch_size * n, self.hid_size * 2).to(x.device)
                    c0 = torch.zeros(2, batch_size * n, self.hid_size * 2).to(x.device)

                    # according to the paper, there is no self-attend 
                    list1 = []
                    for a in range(batch_size * n):
                        list2 = [torch.cat([hidden_state[a], hidden_state[b]]) for b in range(batch_size * n) if b != a]
                        list1.append(torch.stack(list2))
                    # hard_attn_input: size of (N-1) x N x (hid_size*2)
                    hard_attn_input = torch.stack(list1, dim=1)
                    # hard_attn_output: size of (N-1) x N x 2, the third dimension is one-hot vector
                    hard_attn_output = self.lstm(hard_attn_input, (h0, c0))[0]
                    hard_attn_output = self.linear(hard_attn_output)
                    hard_attn_output = F.gumbel_softmax(hard_attn_output, hard=True) 
                    # hard_attn_output: size of (N-1) x N x 1
                    hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
                    # hard_attn_output: size of N x (N-1)
                    hard_attn_output = hard_attn_output.permute(1, 0, 2).squeeze()
                else:
                    hard_attn_output = self.get_hard_attn_one(agent_mask).to(x.device)

                comm_density1 = hard_attn_output.nonzero().size(0) / (batch_size * n * n)
                comm_density2 = hard_attn_output.nonzero().size(0) / (batch_size * n * (n - 1))

                # calculate query and key for soft attention
                q = self.wq(hidden_state)
                k = self.wk(hidden_state)
                
                # Attention computation (batch-wise to avoid cross-episode communication)
                q = q.view(batch_size, n, -1)
                k = k.view(batch_size, n, -1)
                
                # size of batch_size x N x N
                soft_attn = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.qk_hid_size)
                
                # size of batch_size x N x (N-1)
                # Removing diagonal for self-attention
                list_attn = []
                for b in range(batch_size):
                    list_attn.append(torch.stack([torch.cat([soft_attn[b, l][:l], soft_attn[b, l][l+1:]]) for l in range(n)]))
                soft_attn = torch.stack(list_attn)
                
                soft_attn = soft_attn * hard_attn_output.unsqueeze(0)
                soft_attn = F.softmax(soft_attn, dim=-1)
                attn = soft_attn * hard_attn_output.unsqueeze(0)

                # Choose current or prev depending on recurrent
                comm_in = hidden_state.view(batch_size, n, self.hid_size)
                # Mask by agent_mask (n, 1)
                comm_in = comm_in * agent_mask.transpose(0, 1).unsqueeze(-1)
                
                # Compute weighted sum
                # Size of batch_size x N x (hid_size)
                list_comm = []
                for b in range(batch_size):
                    curr_batch_comm = comm_in[b]
                    list_comm.append(torch.stack([(attn[b, l].reshape(n-1, 1) * torch.cat([curr_batch_comm[:l], curr_batch_comm[l+1:]], dim=0)).sum(dim=0) for l in range(n)], dim=0))
                comm = torch.stack(list_comm)
                
                comm = comm.view(batch_size, n, self.hid_size)
                comm = comm * agent_mask.transpose(0, 1).unsqueeze(-1)
                
                if "log_comm" in info and info.get("log_comm", False):
                    info["comm_log"].append(comm.detach().cpu())
                    info["hard_graph_log"].append(hard_attn_output.detach().cpu())
                    info["attn_log"].append(attn.detach().cpu())

                # Update c for the next pass
                c = self.C_modules[i](comm.view(batch_size * n, self.hid_size))
            else:
                comm = torch.zeros(batch_size, n, self.hid_size).to(x.device)
                c = torch.zeros(batch_size * n, self.hid_size).to(x.device)
                comm_density1 = 0
                comm_density2 = 0

        value_head = self.value_head(torch.cat((hidden_state.view(batch_size * n, -1), comm.view(batch_size * n, -1)), dim=-1))
        h = hidden_state.view(batch_size, n, self.hid_size)

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            # comm already has size (batch_size, n, hid_size)
            action = [F.log_softmax(head(torch.cat((h, comm), dim=-1)), dim=-1) for head in self.heads]

        if self.args.recurrent:
            return action, value_head, (hidden_state.view(batch_size * n, -1).clone(), cell_state.clone()), [comm_density1, comm_density2]
        else:
            return action, value_head, [comm_density1, comm_density2]


    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))
    
    def get_hard_attn_one(self, agent_mask):
        n = self.args.nagents
        adj = torch.ones(n, n)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = adj * agent_mask * agent_mask_transpose
        hard_attn = torch.stack([torch.cat([adj[x][:x], adj[x][x+1:n]]) for x in range(n)])

        return hard_attn
