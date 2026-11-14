"""
evaluate_all.py  –  Evaluate a trained MAGIC / baseline model on Traffic Junction.
Computes: Success Rate, IEI (Info Entropy Index), SEI (Specialization Efficiency Index),
          TEI (Topology Efficiency Index) as defined in the MAGIC paper Section IV.

Usage (run from the MAGIC root directory):
  # MAGIC
  python evaluate_all.py --model magic \
      --load saved/traffic_junction/run1/model.pt \
      --num_eval_episodes 100 \
      --nagents 5 --dim 10 --difficulty easy --vision 1 \
      --max_steps 20 --add_rate_min 0.3 --add_rate_max 0.3 \
      --hid_size 128 --first_graph_complete

  # CommNet
  python evaluate_all.py --model commnet \
      --load baselines/saved/traffic_junction/commnet/run1/model.pt \
      --num_eval_episodes 100 \
      --nagents 5 --dim 10 --difficulty easy --vision 1 \
      --max_steps 20 --add_rate_min 0.3 --add_rate_max 0.3 --hid_size 128

  # IC3Net / TarMAC / GAComm: same as CommNet but change --model and --load.
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from inspect import getfullargspec as getargspec

# ── project imports ────────────────────────────────────────────────────────────
MAGIC_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MAGIC_ROOT)
sys.path.insert(0, os.path.join(MAGIC_ROOT, "baselines"))

import data
from utils import init_args_for_env, merge_stat
from action_utils import parse_action_args, select_action, translate_action

torch.set_default_dtype(torch.float64)

# ── argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate trained MARL models on Traffic Junction")

parser.add_argument("--model", required=True,
                    choices=["magic", "commnet", "ic3net", "tarmac", "gacomm"],
                    help="Model type to evaluate")
parser.add_argument("--load", required=True, type=str, help="Path to model.pt checkpoint")
parser.add_argument("--num_eval_episodes", type=int, default=100,
                    help="Number of full episodes to evaluate over (default: 100)")

# ── Traffic Junction env args ──────────────────────────────────────────────────
parser.add_argument("--nagents",        type=int,   default=5)
parser.add_argument("--dim",            type=int,   default=10)
parser.add_argument("--difficulty",     type=str,   default="easy")
parser.add_argument("--vision",         type=int,   default=1)
parser.add_argument("--max_steps",      type=int,   default=20)
parser.add_argument("--add_rate_min",   type=float, default=0.3)
parser.add_argument("--add_rate_max",   type=float, default=0.3)
parser.add_argument("--curr_start",     type=int,   default=0)
parser.add_argument("--curr_end",       type=int,   default=0)
parser.add_argument("--vocab_type",     type=str,   default="bool")
parser.add_argument("--seed",           type=int,   default=42)

# ── Model architecture args (must match training) ─────────────────────────────
parser.add_argument("--hid_size",       type=int,   default=128)

# CommNet / IC3Net / TarMAC / GAComm specific
parser.add_argument("--qk_hid_size",   type=int,   default=16)
parser.add_argument("--value_hid_size", type=int,  default=32)
parser.add_argument("--comm_passes",    type=int,   default=1)
parser.add_argument("--comm_mode",      type=str,   default="avg")
parser.add_argument("--comm_init",      type=str,   default="uniform")
parser.add_argument("--detach_gap",     type=int,   default=10000)
parser.add_argument("--rnn_type",       type=str,   default="MLP")
parser.add_argument("--mean_ratio",     type=float, default=1.0)
parser.add_argument("--share_weights",  action="store_true", default=False)

# MAGIC specific
parser.add_argument("--directed",              action="store_true", default=False)
parser.add_argument("--self_loop_type1",        type=int, default=2)
parser.add_argument("--self_loop_type2",        type=int, default=2)
parser.add_argument("--gat_num_heads",          type=int, default=1)
parser.add_argument("--gat_num_heads_out",      type=int, default=1)
parser.add_argument("--gat_hid_size",           type=int, default=64)
parser.add_argument("--ge_num_heads",           type=int, default=4)
parser.add_argument("--first_gat_normalize",    action="store_true", default=False)
parser.add_argument("--second_gat_normalize",   action="store_true", default=False)
parser.add_argument("--gat_encoder_normalize",  action="store_true", default=False)
parser.add_argument("--use_gat_encoder",        action="store_true", default=False)
parser.add_argument("--gat_encoder_out_size",   type=int, default=64)
parser.add_argument("--first_graph_complete",   action="store_true", default=False)
parser.add_argument("--second_graph_complete",  action="store_true", default=False)
parser.add_argument("--learn_second_graph",     action="store_true", default=False)
parser.add_argument("--message_encoder",        action="store_true", default=False)
parser.add_argument("--message_decoder",        action="store_true", default=False)
parser.add_argument("--comm_mask_zero",         action="store_true", default=False)
parser.add_argument("--advantages_per_action",  action="store_true", default=False)

# Required but unused during pure eval
parser.add_argument("--action_scale",   type=float, default=1.0)
parser.add_argument("--nactions",       type=str,   default="1")
parser.add_argument("--gamma",          type=float, default=1.0)
parser.add_argument("--lrate",          type=float, default=0.001)
parser.add_argument("--value_coeff",    type=float, default=0.01)
parser.add_argument("--entr",           type=float, default=0)
parser.add_argument("--normalize_rewards", action="store_true", default=False)

init_args_for_env(parser)
args = parser.parse_args()

# ── Fixed args for evaluation ─────────────────────────────────────────────────
args.env_name       = "traffic_junction"
args.nprocesses     = 1
args.batch_size     = 1
args.nfriendly      = args.nagents
args.display        = False
args.random         = False
args.recurrent      = False
args.hard_attn      = False
args.comm_action_one = False
args.continuous     = False
args.commnet        = False
args.ic3net         = False
args.tarcomm        = False
args.gacomm         = False

# ── Model-specific flag derivations ───────────────────────────────────────────
if args.model == "ic3net":
    args.commnet         = True
    args.ic3net          = True
    args.hard_attn       = True
    args.mean_ratio      = 0
    args.comm_action_one = True
elif args.model == "tarmac":
    args.commnet         = True
    args.ic3net          = True
    args.tarcomm         = True
    args.hard_attn       = True
    args.mean_ratio      = 0
    args.comm_action_one = True
elif args.model == "gacomm":
    args.commnet         = True
    args.gacomm          = True
    args.mean_ratio      = 0
    args.comm_action_one = True
elif args.model == "commnet":
    args.commnet         = True
elif args.model == "magic":
    args.first_graph_complete = True

if args.commnet and args.rnn_type == "LSTM":
    args.recurrent = True

if args.seed != -1:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

# ── Build environment ──────────────────────────────────────────────────────────
env = data.init(args.env_name, args, False)
args.obs_size    = env.observation_dim
args.num_actions = env.num_actions
if not isinstance(args.num_actions, (list, tuple)):
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs  = env.observation_dim

if args.hard_attn and args.commnet:
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

if args.commnet and (getattr(args, "recurrent", False) or args.rnn_type == "LSTM"):
    args.recurrent = True
    args.rnn_type  = "LSTM"

parse_action_args(args)

num_inputs = env.observation_dim

# ── Construct policy network ───────────────────────────────────────────────────
if args.model == "magic":
    from magic import MAGIC
    policy_net = MAGIC(args)
elif args.model == "gacomm":
    from baselines.ga_comm import GACommNetMLP
    policy_net = GACommNetMLP(args, num_inputs)
elif args.model == "tarmac":
    from baselines.tar_comm import TarCommNetMLP
    policy_net = TarCommNetMLP(args, num_inputs)
else:
    from baselines.comm import CommNetMLP
    policy_net = CommNetMLP(args, num_inputs)

# ── Load checkpoint ────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Model      : {args.model.upper()}")
print(f"  Checkpoint : {args.load}")
print(f"{'='*70}\n")

ckpt = torch.load(args.load, map_location="cpu", weights_only=False)
policy_net.load_state_dict(ckpt["policy_net"])
policy_net.eval()


# ═══════════════════════════════════════════════════════════════════════════════
#  METRIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def message_entropy(msg_tensor):
    """
    Compute Shannon entropy of a message vector for one agent per round.
    msg: [hid_size]  — treated as an un-normalised probability distribution via softmax.
    H(m) = -sum_k p_k * log2(p_k)
    """
    msg = msg_tensor.float()
    # Use softmax to convert activations into a prob distribution
    p = F.softmax(msg, dim=-1)
    # clamp to avoid log(0)
    p = p.clamp(min=1e-9)
    return -(p * torch.log2(p)).sum().item()


def cosine_similarity_matrix(msgs):
    """
    msgs: [N, hid_size]
    Returns mean pairwise cosine similarity (off-diagonal upper triangle).
    """
    N = msgs.shape[0]
    if N < 2:
        return 0.0
    norm = msgs.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    msgs_n = msgs / norm
    sim_matrix = torch.mm(msgs_n.float(), msgs_n.float().T)  # [N, N]
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    if not pairs:
        return 0.0
    return sum(sim_matrix[i, j].item() for i, j in pairs) / len(pairs)


# ── Hook infrastructure ────────────────────────────────────────────────────────
# We capture messages (hidden state vectors used as comm) and graph (adjacency)
# per forward() call using a simple storage dict.
_captured = {}

def install_hooks(model, model_type, n_agents):
    """
    Install forward hooks on the model to capture messages and comm graph G
    each time forward() is called (i.e., each timestep).

    Captured per timestep (list per call):
        _captured['msgs']   : list of tensors [N, hid_size] per round L
        _captured['graphs'] : list of tensors [N, N] per round L
    """
    handles = []

    if model_type == "magic":
        # MAGIC has two GAT sub-processors. Hook their outputs to get messages
        # after each communication round.
        round_msgs = []
        round_graphs = []

        def hook_sp1(module, inp, out):
            # out: [N, gat_hid_size * num_heads]  (after sub_processor1)
            round_msgs.clear()
            round_graphs.clear()
            round_msgs.append(out.detach())            # round 1 messages

        def hook_sp2(module, inp, out):
            round_msgs.append(out.detach())            # round 2 messages

        def hook_forward(module, inp, out):
            # Capture the adj matrices built inside forward()
            # We reconstruct them from the model state after the call
            # Since adj1/adj2 are local vars, we patch forward() instead.
            pass

        # Instead of hooking sub-processors for graphs (they don't see adj),
        # we monkey-patch forward() to expose adj1/adj2.
        original_forward = model.forward

        def patched_forward(x, info={}):
            obs, extras = x
            encoded_obs   = model.obs_encoder(obs)
            hidden_state, cell_state = extras
            batch_size = encoded_obs.size()[0]

            num_agents_alive, agent_mask = model.get_agent_mask(batch_size, info)
            if model.args.comm_mask_zero:
                agent_mask *= torch.zeros(n_agents, 1)

            hidden_state, cell_state = model.lstm_cell(
                encoded_obs.squeeze(), (hidden_state, cell_state))

            comm = hidden_state
            if model.args.message_encoder:
                comm = model.message_encoder(comm)
            comm = comm * agent_mask
            comm_ori = comm.clone()

            # sub-scheduler 1 / adj1
            if not model.args.first_graph_complete:
                if model.args.use_gat_encoder:
                    adj_complete = model.get_complete_graph(agent_mask)
                    encoded_state1 = model.gat_encoder(comm, adj_complete)
                    adj1 = model.sub_scheduler(model.sub_scheduler_mlp1, encoded_state1, agent_mask, model.args.directed)
                else:
                    adj1 = model.sub_scheduler(model.sub_scheduler_mlp1, comm, agent_mask, model.args.directed)
            else:
                adj1 = model.get_complete_graph(agent_mask)

            comm = F.elu(model.sub_processor1(comm, adj1))

            # sub-scheduler 2 / adj2
            if model.args.learn_second_graph and not model.args.second_graph_complete:
                if model.args.use_gat_encoder:
                    if model.args.first_graph_complete:
                        adj_complete = model.get_complete_graph(agent_mask)
                        encoded_state2 = model.gat_encoder(comm_ori, adj_complete)
                    else:
                        encoded_state2 = encoded_state1
                    adj2 = model.sub_scheduler(model.sub_scheduler_mlp2, encoded_state2, agent_mask, model.args.directed)
                else:
                    adj2 = model.sub_scheduler(model.sub_scheduler_mlp2, comm_ori, agent_mask, model.args.directed)
            elif not model.args.learn_second_graph and not model.args.second_graph_complete:
                adj2 = adj1
            else:
                adj2 = model.get_complete_graph(agent_mask)

            comm_r1 = comm.clone()   # messages after round 1 (shape: [N, gat_hid_size*heads])
            comm = model.sub_processor2(comm, adj2)
            comm = comm * agent_mask
            if model.args.message_decoder:
                comm = model.message_decoder(comm)

            # ── store captured tensors ──
            # Round 1: comm_r1 has dim gat_hid_size*heads; round 2: comm has hid_size.
            # For consistent IEI/SEI we use the hidden_state (pre-GAT) as round-1 message
            # and comm (post round-2 GAT) as round-2 message — both are [N, hid_size].
            _captured['msgs']   = [hidden_state.detach(), comm.detach()]   # L=2 rounds, [N, hid_size]
            _captured['graphs'] = [adj1.detach(), adj2.detach()]            # L=2 rounds, [N, N]

            value_head = model.value_head(
                torch.cat((hidden_state, comm), dim=-1))
            h = hidden_state.view(batch_size, n_agents, model.hid_size)
            c = comm.view(batch_size, n_agents, model.hid_size)
            action_out = [F.log_softmax(ah(torch.cat((h, c), dim=-1)), dim=-1)
                          for ah in model.action_heads]
            return action_out, value_head, (hidden_state.clone(), cell_state.clone())

        model.forward = patched_forward

    elif model_type == "tarmac":
        # TarMAC: hook the attention output (the weighted-value sum = effective message)
        # and construct a soft-thresholded G from attn weights.
        original_forward = model.forward

        def patched_forward_tarmac(x, info={}):
            x_enc, hidden_state, cell_state = model.forward_state_encoder(x)
            batch_size = x_enc.size()[0]
            num_agents_alive, agent_mask = model.get_agent_mask(batch_size, info)
            agent_mask_alive = agent_mask.clone()

            if model.args.hard_attn:
                comm_action = torch.tensor(info['comm_action'])
                comm_action_mask = comm_action.expand(batch_size, n_agents, n_agents).unsqueeze(-1)
                agent_mask *= comm_action_mask.double()

            agent_mask_transpose = agent_mask.transpose(1, 2)

            all_msgs, all_graphs = [], []
            for i in range(model.comm_passes):
                comm = hidden_state.view(batch_size, n_agents, model.hid_size) \
                    if model.args.recurrent else hidden_state

                query = model.state2query(comm)   # [B, N, 16]
                key   = model.state2key(comm)
                value = model.state2value(comm)   # [B, N, hid_size]

                import math
                scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(model.hid_size)
                scores = scores.masked_fill(agent_mask.squeeze(-1) == 0, -1e9)
                attn   = F.softmax(scores, dim=-1)
                attn   = attn * agent_mask.squeeze(-1)

                comm_out = torch.matmul(attn, value)   # [B, N, hid_size]
                comm_out *= agent_mask_alive.squeeze(-1)[:, 0].unsqueeze(-1).expand(batch_size, n_agents, model.hid_size)

                c = model.C_modules[i](comm_out)

                if model.args.recurrent:
                    inp = x_enc + c
                    inp = inp.view(batch_size * n_agents, model.hid_size)
                    output = model.f_module(inp, (hidden_state, cell_state))
                    hidden_state = output[0]
                    cell_state   = output[1]
                else:
                    hidden_state = sum([x_enc, model.f_modules[i](hidden_state), c])
                    hidden_state = model.tanh(hidden_state)

                # comm_out [B, N, hid_size] → [N, hid_size] (batch=1)
                all_msgs.append(comm_out.squeeze(0).detach())
                # G: threshold attn at > 0 → binary [N, N]
                G = (attn.squeeze(0) > 1e-6).float().detach()
                all_graphs.append(G)

            _captured['msgs']   = all_msgs
            _captured['graphs'] = all_graphs

            value_head = model.value_head(hidden_state)
            h = hidden_state.view(batch_size, n_agents, model.hid_size)
            action = [F.log_softmax(head(h), dim=-1) for head in model.heads]

            if model.args.recurrent:
                return action, value_head, (hidden_state.clone(), cell_state.clone())
            else:
                return action, value_head

        model.forward = patched_forward_tarmac

    elif model_type == "gacomm":
        original_forward = model.forward

        def patched_forward_gacomm(x, info={}):
            x_enc, hidden_state, cell_state = model.forward_state_encoder(x)
            batch_size = x_enc.size()[0]
            num_agents_alive, agent_mask = model.get_agent_mask(batch_size, info)

            c = torch.zeros(batch_size * n_agents, model.hid_size).to(x_enc.device)
            comm_density1 = comm_density2 = 0
            all_msgs, all_graphs = [], []

            for i in range(model.comm_passes):
                if model.args.recurrent:
                    inp = x_enc + c
                    inp = inp.view(batch_size * n_agents, model.hid_size)
                    output = model.f_module(inp, (hidden_state, cell_state))
                    hidden_state = output[0]
                    cell_state   = output[1]
                else:
                    hidden_state = sum([x_enc, model.f_modules[i](hidden_state), c])
                    hidden_state = model.tanh(hidden_state)

                if not model.args.comm_mask_zero:
                    if not model.args.comm_action_one:
                        h0 = torch.zeros(2, batch_size * n_agents, model.hid_size * 2)
                        c0 = torch.zeros(2, batch_size * n_agents, model.hid_size * 2)
                        list1 = []
                        for a in range(batch_size * n_agents):
                            list2 = [torch.cat([hidden_state[a], hidden_state[b]])
                                     for b in range(batch_size * n_agents) if b != a]
                            list1.append(torch.stack(list2))
                        hard_attn_input  = torch.stack(list1, dim=1)
                        hard_attn_output = model.lstm(hard_attn_input, (h0, c0))[0]
                        hard_attn_output = model.linear(hard_attn_output)
                        hard_attn_output = F.gumbel_softmax(hard_attn_output, hard=True)
                        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
                        hard_attn_output = hard_attn_output.permute(1, 0, 2).squeeze()
                    else:
                        hard_attn_output = model.get_hard_attn_one(agent_mask)

                    comm_density1 = hard_attn_output.nonzero().size(0) / (batch_size * n_agents * n_agents)
                    comm_density2 = hard_attn_output.nonzero().size(0) / (batch_size * n_agents * (n_agents - 1))

                    q = model.wq(hidden_state).view(batch_size, n_agents, -1)
                    k = model.wk(hidden_state).view(batch_size, n_agents, -1)
                    soft_attn = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(model.qk_hid_size)
                    list_attn = []
                    for b in range(batch_size):
                        list_attn.append(torch.stack([
                            torch.cat([soft_attn[b, l][:l], soft_attn[b, l][l+1:]])
                            for l in range(n_agents)]))
                    soft_attn = torch.stack(list_attn)
                    soft_attn = soft_attn * hard_attn_output.unsqueeze(0)
                    soft_attn = F.softmax(soft_attn, dim=-1)
                    attn = soft_attn * hard_attn_output.unsqueeze(0)

                    comm_in = hidden_state.view(batch_size, n_agents, model.hid_size)
                    comm_in = comm_in * agent_mask.transpose(0, 1).unsqueeze(-1)
                    list_comm = []
                    for b in range(batch_size):
                        curr = comm_in[b]
                        list_comm.append(torch.stack([
                            (attn[b, l].reshape(n_agents-1, 1) *
                             torch.cat([curr[:l], curr[l+1:]], dim=0)).sum(dim=0)
                            for l in range(n_agents)]))
                    comm = torch.stack(list_comm)
                    comm = comm.view(batch_size, n_agents, model.hid_size)
                    comm = comm * agent_mask.transpose(0, 1).unsqueeze(-1)
                    c = model.C_modules[i](comm.view(batch_size * n_agents, model.hid_size))

                    # Build N×N G from hard_attn_output (N×N-1) → insert 0 on diagonal
                    G_full = torch.zeros(n_agents, n_agents, dtype=torch.float64)
                    for row in range(n_agents):
                        cols = list(range(n_agents))
                        cols.pop(row)
                        G_full[row, cols] = hard_attn_output[row].double().detach()
                    all_msgs.append(comm.squeeze(0).detach())
                    all_graphs.append(G_full)
                else:
                    comm = torch.zeros(batch_size, n_agents, model.hid_size)
                    c    = torch.zeros(batch_size * n_agents, model.hid_size)
                    all_msgs.append(comm.squeeze(0).detach())
                    all_graphs.append(torch.zeros(n_agents, n_agents))

            _captured['msgs']   = all_msgs
            _captured['graphs'] = all_graphs

            value_head = model.value_head(
                torch.cat((hidden_state.view(batch_size * n_agents, -1),
                           comm.view(batch_size * n_agents, -1)), dim=-1))
            h = hidden_state.view(batch_size, n_agents, model.hid_size)
            action = [F.log_softmax(head(torch.cat((h, comm), dim=-1)), dim=-1)
                      for head in model.heads]

            if model.args.recurrent:
                return action, value_head, \
                    (hidden_state.view(batch_size * n_agents, -1).clone(), cell_state.clone()), \
                    [comm_density1, comm_density2]
            else:
                return action, value_head, [comm_density1, comm_density2]

        model.forward = patched_forward_gacomm

    else:
        # CommNet / IC3Net — hook the final hidden state (used as messages),
        # graph G = comm_mask (all-one minus identity; for IC3Net with comm_action_one=True).
        original_forward = model.forward

        def patched_forward_commnet(x, info={}):
            x_enc, hidden_state, cell_state = model.forward_state_encoder(x)
            batch_size = x_enc.size()[0]
            num_agents_alive, agent_mask = model.get_agent_mask(batch_size, info)

            if model.args.hard_attn:
                comm_action = torch.tensor(info['comm_action'])
                comm_action_mask = comm_action.expand(batch_size, n_agents, n_agents).unsqueeze(-1)
                agent_mask *= comm_action_mask.double()

            agent_mask_transpose = agent_mask.transpose(1, 2)

            all_msgs, all_graphs = [], []
            for i in range(model.comm_passes):
                comm = hidden_state.view(batch_size, n_agents, model.hid_size) \
                    if model.args.recurrent else hidden_state
                comm_exp = comm.unsqueeze(-2).expand(-1, n_agents, n_agents, model.hid_size)

                mask = model.comm_mask.view(1, n_agents, n_agents).expand(comm_exp.shape[0], n_agents, n_agents).unsqueeze(-1)
                mask = mask.expand_as(comm_exp)
                comm_exp = comm_exp * mask

                if hasattr(model.args, 'comm_mode') and model.args.comm_mode == 'avg' and num_agents_alive > 1:
                    comm_exp = comm_exp / (num_agents_alive - 1)

                comm_exp = comm_exp * agent_mask
                comm_exp = comm_exp * agent_mask_transpose

                # For messages: each agent's hidden state is its "message"
                msgs = comm.squeeze(0).detach()   # [N, hid_size]
                all_msgs.append(msgs)

                # G: who sends to whom = comm_mask ∩ agent_mask (binary [N, N])
                G = (model.comm_mask * agent_mask.squeeze(0).squeeze(-1).expand(n_agents, n_agents)).detach()
                all_graphs.append(G)

                comm_sum = comm_exp.sum(dim=1)
                c = model.C_modules[i](comm_sum)

                if model.args.recurrent:
                    inp = x_enc + c
                    inp = inp.view(batch_size * n_agents, model.hid_size)
                    output = model.f_module(inp, (hidden_state, cell_state))
                    hidden_state = output[0]
                    cell_state   = output[1]
                else:
                    hidden_state = sum([x_enc, model.f_modules[i](hidden_state), c])
                    hidden_state = model.tanh(hidden_state)

            _captured['msgs']   = all_msgs
            _captured['graphs'] = all_graphs

            value_head = model.value_head(hidden_state)
            h = hidden_state.view(batch_size, n_agents, model.hid_size)
            action = [F.log_softmax(head(h), dim=-1) for head in model.heads]

            if model.args.recurrent:
                return action, value_head, (hidden_state.clone(), cell_state.clone())
            else:
                return action, value_head

        model.forward = patched_forward_commnet

    return handles


def compute_step_metrics(n_agents):
    """
    From _captured['msgs'] and _captured['graphs'] for ONE timestep t,
    compute the per-timestep quantities defined in the MAGIC paper:

        H_t_step  = (1/L) * sum_l [ (1/N) * sum_i H(m_i^l) ]
        xi_t_step = (1/L) * sum_l [ (2/(N*(N-1))) * sum_{i<j} cos(m_i^l, m_j^l) ]
        C_t_step  = sum_l  sum_i sum_j G_{ij}^l

    These are later averaged across all timesteps of all episodes.
    """
    msgs   = _captured.get('msgs',   [])
    graphs = _captured.get('graphs', [])
    L      = len(msgs)

    if L == 0:
        return 0.0, 0.0, 0.0

    H_sum   = 0.0
    xi_sum  = 0.0
    C_t     = 0.0

    for l in range(L):
        m = msgs[l]   # expected [N, hid_size]
        if m.dim() == 1:
            m = m.unsqueeze(0)
        # Defensive: trim to n_agents rows if extra rows present
        if m.shape[0] > n_agents:
            m = m[:n_agents]
        N_alive = m.shape[0]  # may be < n_agents if agents are dead

        # ── IEI contribution: (1/N) * sum_i H(m_i)  ──────────────────────────
        # H(m_i) = -sum_k p_k * log2(p_k);  p = softmax(m_i)  (treats activations as logits)
        H_l = sum(message_entropy(m[i]) for i in range(N_alive)) / max(N_alive, 1)
        H_sum += H_l

        # ── SEI contribution: (2/(N*(N-1))) * sum_{i<j} cos(m_i, m_j)  ───────
        # Equals mean of upper-triangle cosine similarities
        xi_l = cosine_similarity_matrix(m)   # already = mean of upper triangle
        xi_sum += xi_l

        # ── TEI contribution: total edges in G  ───────────────────────────────
        G = graphs[l]
        C_t += G.sum().item()

    # Average over L communication rounds (paper formula: 1/L * sum_l)
    H_t_step  = H_sum  / L
    xi_t_step = xi_sum / L
    # C_t is summed (not averaged) over rounds — TEI uses total edge count

    return H_t_step, xi_t_step, C_t


# ── Install hooks ──────────────────────────────────────────────────────────────
install_hooks(policy_net, args.model, args.nagents)

# ── Evaluation loop ────────────────────────────────────────────────────────────
total_stats    = {}
total_episodes = 0

# Accumulators for IEI / SEI / TEI across ALL timesteps in all episodes
all_H   = []   # one value per timestep
all_xi  = []   # one value per timestep
ep_C_list   = []   # one entry per episode: total comm edges in that episode

with torch.no_grad():
    for ep_idx in range(args.num_eval_episodes):
        reset_args_spec = getargspec(env.reset).args
        state = env.reset(0) if "epoch" in reset_args_spec else env.reset()

        info    = {}
        ep_stat = {}

        if args.hard_attn and args.commnet:
            info["comm_action"] = np.zeros(args.nagents, dtype=int)

        prev_hid = torch.zeros(1, args.nagents, args.hid_size)
        ep_C = 0.0

        for t in range(args.max_steps):
            _captured.clear()

            if args.model == "magic":
                if t == 0:
                    prev_hid = policy_net.init_hidden(batch_size=state.shape[0])
                x = [state, prev_hid]
                action_out, value, prev_hid = policy_net(x, info)

            elif args.recurrent:
                if args.rnn_type == "LSTM" and t == 0:
                    prev_hid = policy_net.init_hidden(batch_size=state.shape[0])
                x = [state, prev_hid]
                if args.gacomm:
                    action_out, value, prev_hid, comm_density = policy_net(x, info)
                else:
                    action_out, value, prev_hid = policy_net(x, info)

                if (t + 1) % args.detach_gap == 0:
                    if args.rnn_type == "LSTM":
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()

            else:
                x = state
                if args.gacomm:
                    action_out, value, comm_density = policy_net(x, info)
                else:
                    action_out, value = policy_net(x, info)

            # ── Collect per-timestep metric contributions ──────────────────────
            H_step, xi_step, C_step = compute_step_metrics(args.nagents)
            all_H.append(H_step)
            all_xi.append(xi_step)
            ep_C += C_step

            # ── Step the environment ───────────────────────────────────────────
            action         = select_action(args, action_out)
            action, actual = translate_action(args, env, action)
            next_state, reward, done_flag, info = env.step(actual)

            if args.hard_attn and args.commnet:
                info["comm_action"] = (
                    action[-1] if not args.comm_action_one
                    else np.ones(args.nagents, dtype=int)
                )
                ep_stat["comm_action"] = (
                    ep_stat.get("comm_action", 0) + info["comm_action"][:args.nfriendly]
                )

            ep_stat["reward"] = ep_stat.get("reward", 0) + reward[:args.nfriendly]
            done  = done_flag or (t == args.max_steps - 1)
            state = next_state
            if done:
                break

        ep_C_list.append(ep_C)

        ep_stat["num_steps"]   = t + 1
        ep_stat["steps_taken"] = t + 1

        if hasattr(env, "reward_terminal"):
            extra_r = env.reward_terminal()
            ep_stat["reward"] = ep_stat.get("reward", 0) + extra_r[:args.nfriendly]

        if hasattr(env, "get_stat"):
            merge_stat(env.get_stat(), ep_stat)

        merge_stat(ep_stat, total_stats)
        total_episodes += 1
        
        ep_success = ep_stat.get("success", 0)
        cum_success = total_stats.get("success", 0)
        print(f"[{args.model.upper()}] Episode {total_episodes} | Success: {ep_success} | Cum_Success_Rate: {cum_success/total_episodes:.4f}")



# ═══════════════════════════════════════════════════════════════════════════════
#  COMPUTE FINAL METRICS  (Paper: Section IV)
# ═══════════════════════════════════════════════════════════════════════════════

N          = total_episodes
S_count    = total_stats.get("success", 0)        # raw number of successes
S_t        = S_count / max(N, 1)                  # SUCCESS RATE (fraction 0-1), paper's S_t

# H_bar: average over ALL timesteps of all evaluated episodes
#   H_step = (1/L) * sum_l [ (1/N) * sum_i H(m_i^l) ]  (paper formula)
H_bar      = float(np.mean(all_H))  if all_H  else 0.0

# xi_bar: average over ALL timesteps of all evaluated episodes
#   xi_step = (1/L) * sum_l [ (2/(N*(N-1))) * sum_{i<j} cos(m_i^l, m_j^l) ]
xi_bar     = float(np.mean(all_xi)) if all_xi else 0.0

# C_avg: average total comm edges per episode
#   C_t = sum_l sum_i sum_j G_{ij}^l  (summed across all timesteps in one episode)
C_avg      = float(np.mean(ep_C_list))  if ep_C_list  else 0.0
C_total    = float(np.sum(ep_C_list))

# ── Final metric values ────────────────────────────────────────────────────────
# IEI: more entropy per unit success = model needs more bits of info to succeed → bad
IEI = H_bar  / (S_t + 1e-9)    # lower is better

# SEI: more message similarity per unit success = agents send redundant msgs → bad
SEI = xi_bar / (S_t + 1e-9)    # lower is better

# TEI: success per unit of comm edges used = efficient topology → good
#   Uses C_avg (per-episode) so the metric is scale-invariant w.r.t. episode count
TEI = S_t    / (C_avg + 1e-9)  # higher is better

# ── Print results ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  EVALUATION RESULTS  |  {args.model.upper()}  |  {N} episodes")
print(f"{'='*70}")
avg_reward = np.sum(total_stats.get("reward", [0])) / N
print(f"  Avg Reward/agent    : {avg_reward / args.nfriendly:.4f}")
print(f"  Avg Team Reward     : {avg_reward:.4f}")
if "success" in total_stats:
    print(f"  Success Rate (S_t)  : {S_t:.4f}  ({S_count:.0f}/{N})")
if "steps_taken" in total_stats:
    print(f"  Avg Steps Taken     : {total_stats['steps_taken'] / N:.2f}")
print()
print(f"  ── Efficiency Metrics (Paper Section IV) ──")
print(f"  H_bar   (avg msg entropy)  : {H_bar:.4f}  bits / round / agent")
print(f"  xi_bar  (avg cos-sim)      : {xi_bar:.4f}  per round")
print(f"  C_avg   (avg comm edges)   : {C_avg:.2f}   per episode")
print(f"  C_total (total comm edges) : {C_total:.0f}")
print()
print(f"  IEI  = H_bar  / S_t  : {IEI:.6f}   (↓ lower is better)")
print(f"  SEI  = xi_bar / S_t  : {SEI:.6f}   (↓ lower is better)")
print(f"  TEI  = S_t / C_avg   : {TEI:.6f}   (↑ higher is better)")
print(f"{'='*70}\n")



