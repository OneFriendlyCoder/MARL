import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from action_utils import select_action, translate_action
from gnn_layers import GraphAttention

class MAGIC(nn.Module):
    """
    The communication protocol of Multi-Agent Graph AttentIon Communication (MAGIC)
    Optimized for 1 Round of Communication.
    """
    def __init__(self, args):
        super(MAGIC, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        dropout = 0
        negative_slope = 0.2

        # 1-Round Processor: 
        # We set output size to hid_size and average=True so the output 
        # is immediately compatible with the hidden state for the action head.
        self.sub_processor1 = GraphAttention(
            args.hid_size, 
            args.hid_size, 
            dropout=dropout, 
            negative_slope=negative_slope, 
            num_heads=args.gat_num_heads, 
            self_loop_type=args.self_loop_type1, 
            average=True, 
            normalize=args.first_gat_normalize
        )

        # Encoder for the Scheduler
        if args.use_gat_encoder:
            self.gat_encoder = GraphAttention(
                args.hid_size, 
                args.gat_encoder_out_size, 
                dropout=dropout, 
                negative_slope=negative_slope, 
                num_heads=args.ge_num_heads, 
                self_loop_type=1, 
                average=True, 
                normalize=args.gat_encoder_normalize
            )

        self.obs_encoder = nn.Linear(args.obs_size, args.hid_size)
        self.init_hidden(args.batch_size)
        self.lstm_cell = nn.LSTMCell(args.hid_size, args.hid_size)

        # Sub-scheduler 1 logic
        if not args.first_graph_complete:
            input_size = args.gat_encoder_out_size * 2 if args.use_gat_encoder else self.hid_size * 2
            self.sub_scheduler_mlp1 = nn.Sequential(
                nn.Linear(input_size, self.hid_size // 2),
                nn.ReLU(),
                nn.Linear(self.hid_size // 2, self.hid_size // 8),
                nn.ReLU(),
                nn.Linear(self.hid_size // 8, 2)
            )

        if args.message_encoder:
            self.message_encoder = nn.Linear(args.hid_size, args.hid_size)
        if args.message_decoder:
            self.message_decoder = nn.Linear(args.hid_size, args.hid_size)

        # Weight initialization
        if args.comm_init == 'zeros':
            if args.message_encoder: self.message_encoder.weight.data.zero_()
            if args.message_decoder: self.message_decoder.weight.data.zero_()
        
        if not args.first_graph_complete:
            self.sub_scheduler_mlp1.apply(self.init_linear)

        # Action and Value heads
        self.action_heads = nn.ModuleList([nn.Linear(2 * args.hid_size, o) for o in args.naction_heads])
        self.value_head = nn.Linear(2 * self.hid_size, 1)

    def forward(self, x, info={}):
        obs, extras = x
        encoded_obs = self.obs_encoder(obs)
        hidden_state, cell_state = extras
        batch_size = encoded_obs.size()[0]
        n = self.nagents
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        if self.args.comm_mask_zero:
            agent_mask *= torch.zeros(n, 1)

        hidden_state, cell_state = self.lstm_cell(encoded_obs.squeeze(), (hidden_state, cell_state))
        
        comm = hidden_state
        if self.args.message_encoder:
            comm = self.message_encoder(comm)

        comm = comm * agent_mask
        
        # Round 1 Scheduler
        if not self.args.first_graph_complete:
            if self.args.use_gat_encoder:
                adj_complete = self.get_complete_graph(agent_mask)
                encoded_state1 = self.gat_encoder(comm, adj_complete)
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, encoded_state1, agent_mask, self.args.directed)
            else:
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask, self.args.directed)
        else:
            adj1 = self.get_complete_graph(agent_mask)

        # Round 1 Processor (Final Round)
        comm = F.elu(self.sub_processor1(comm, adj1))
        comm = comm * agent_mask

        if self.args.message_decoder:
            comm = self.message_decoder(comm)

        # Prepare outputs
        value_head = self.value_head(torch.cat((hidden_state, comm), dim=-1))
        h = hidden_state.view(batch_size, n, self.hid_size)
        c = comm.view(batch_size, n, self.hid_size)

        # Logging for 1 Round
        if "log_comm" in info and info.get("log_comm", False):
            info["comm_log"].append(c.detach().cpu())
            # Only log one adjacency matrix per step
            info["hard_graph_log"].append([adj1.detach().cpu()])

        action_out = [F.log_softmax(action_head(torch.cat((h, c), dim=-1)), dim=-1) for action_head in self.action_heads]
        return action_out, value_head, (hidden_state.clone(), cell_state.clone())

    def get_agent_mask(self, batch_size, info):
        n = self.nagents
        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
        else:
            agent_mask = torch.ones(n)
        agent_mask = agent_mask.view(n, 1).clone()
        return agent_mask.sum(), agent_mask

    def init_linear(self, m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)

    def init_hidden(self, batch_size):
        return tuple((torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                      torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

    def sub_scheduler(self, sub_scheduler_mlp, hidden_state, agent_mask, directed=True):
        n = self.args.nagents
        hid_size = hidden_state.size(-1)
        hard_attn_input = torch.cat([hidden_state.repeat(1, n).view(n * n, -1), 
                                   hidden_state.repeat(n, 1)], dim=1).view(n, -1, 2 * hid_size)
        
        if directed:
            hard_attn_output = F.gumbel_softmax(sub_scheduler_mlp(hard_attn_input), hard=True)
        else:
            out = 0.5 * sub_scheduler_mlp(hard_attn_input) + 0.5 * sub_scheduler_mlp(hard_attn_input.permute(1,0,2))
            hard_attn_output = F.gumbel_softmax(out, hard=True)

        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = hard_attn_output.squeeze() * agent_mask * agent_mask_transpose
        return adj

    def get_complete_graph(self, agent_mask):
        n = self.args.nagents
        adj = torch.ones(n, n)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        return adj * agent_mask * agent_mask_transpose