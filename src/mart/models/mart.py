import torch
import torch.nn as nn

import numpy as np

from .prt import RT, RTNoEdgeInit
from .hrt import HRT, HRTNoEdgeInit


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(1024, 512), activation='relu'):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_dims)
        dims.append(output_dim)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class PositionalAgentEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_t_len=200, concat=True):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe[None].repeat(num_a, 1, 1)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset):
        ae = self.ae[a_offset: num_a + a_offset, :]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, t_offset=0):
        num_t = x.shape[1]
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset) #(N, T, D)
        if self.concat:
            feat = [x, pos_enc]
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
        return self.dropout(x) #(N, T, D)


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.multiplier = len(args.hyper_scales) + 1
        # Laplace NLL: output 4 channels per step (loc_x, loc_y, raw_scale_x, raw_scale_y)
        self.laplace = getattr(args, 'laplace_nll', False)
        out_channels = 4 if self.laplace else 2
        self.decoder_mlp = MLP(
            args.model_dim*self.multiplier,
            args.future_length*out_channels,
            hidden_dims=(
                args.decoder_hidden_dim,
                args.decoder_hidden_dim // 2
            )
        )

    def forward(self, final_feature, cur_location):
        outputs = self.decoder_mlp(final_feature)
        if self.laplace:
            outputs = outputs.view(-1, self.args.future_length, 4)
            # Add position offset only to loc channels, not scale
            loc = outputs[..., :2] + cur_location
            raw_scale = outputs[..., 2:]
            return torch.cat([loc, raw_scale], dim=-1)   # [B*N, T_f, 4]
        else:
            outputs = outputs.view(-1, self.args.future_length, 2)
            if not self.args.pred_rel:
                outputs = outputs + cur_location
            return outputs        


def build_edge_type_matrix(use_hoops=False):
    """Build a fixed [N, N] edge-type index matrix from canonical agent layout.

    Canonical order: TeamA(0-4) | TeamB(5-9) | Ball(10) [| Hoop(11-12)]

    Types:
        0  same_team_A       i,j in [0-4], i≠j
        1  same_team_B       i,j in [5-9], i≠j
        2  opponent_A→B      i in [0-4], j in [5-9]
        3  opponent_B→A      i in [5-9], j in [0-4]
        4  player→ball       i in [0-9], j == 10
        5  ball→player       i == 10,    j in [0-9]
        6  self              i == j
        7  agent→hoop        i in [0-10], j in [11-12]   (hoops only)
        8  hoop→agent        i in [11-12], j in [0-10]   (hoops only)
        9  hoop→hoop         i,j in [11-12], i≠j         (hoops only)
    """
    N = 13 if use_hoops else 11
    num_types = 10 if use_hoops else 7

    TEAM_A = set(range(5))
    TEAM_B = set(range(5, 10))
    BALL   = {10}
    HOOPS  = {11, 12} if use_hoops else set()
    REAL   = TEAM_A | TEAM_B | BALL

    mat = torch.zeros(N, N, dtype=torch.long)
    for i in range(N):
        for j in range(N):
            if i == j:
                mat[i, j] = 6
            elif i in TEAM_A and j in TEAM_A:
                mat[i, j] = 0
            elif i in TEAM_B and j in TEAM_B:
                mat[i, j] = 1
            elif i in TEAM_A and j in TEAM_B:
                mat[i, j] = 2
            elif i in TEAM_B and j in TEAM_A:
                mat[i, j] = 3
            elif i in REAL and j in BALL:
                mat[i, j] = 4
            elif i in BALL and j in REAL:
                mat[i, j] = 5
            elif j in HOOPS:
                mat[i, j] = 7
            elif i in HOOPS:
                mat[i, j] = 8  # covers hoop→hoop handled below
            # hoop→hoop (i≠j both in HOOPS): caught by i in HOOPS above → 8
            # but we want a distinct type, so override:
            if i in HOOPS and j in HOOPS and i != j:
                mat[i, j] = 9

    return mat, num_types


class MART(nn.Module):
    def __init__(self, args):
        super(MART, self).__init__()
        self.args = args

        module_args = {
            'num_layers': 1,
            'num_heads': args.num_heads,
            'node_dim': args.model_dim,
            'node_hidden_dim': args.hidden_dim,
            'edge_dim': args.model_dim,
            'edge_hidden_dim_1': args.hidden_dim,
            'edge_hidden_dim_2': args.hidden_dim,
            'dropout': args.dropout,
        }
        
        self.input_dim = len(args.inputs) + getattr(args, 'extra_input_dim', 0)
        self.input_fc = nn.Linear(self.input_dim, args.model_dim)
        self.input_fc2 = nn.Linear(args.model_dim*args.past_length, args.model_dim)
        
        self.pos_encoder = PositionalAgentEncoding(args.model_dim, 0.1, concat=True)
        
        self.pair_encoders = nn.ModuleList()
        self.hyper_encoders = nn.ModuleList()
        
        # Build the fixed edge-type matrix from canonical NBA agent layout and
        # inject it into the first pair encoder (the one that initialises edges).
        # Only activated when opts.edge_type_emb=True so existing runs are unaffected.
        if getattr(args, 'edge_type_emb', False):
            use_hoops = getattr(args, 'use_hoops', False)
            edge_type_matrix, num_edge_types = build_edge_type_matrix(use_hoops)
        else:
            edge_type_matrix, num_edge_types = None, 0

        for i in range(args.num_layers):
            if i == 0:
                self.pair_encoders.append(RT(
                    **module_args,
                    edge_type_matrix=edge_type_matrix,
                    num_edge_types=num_edge_types,
                ))
            else:
                self.pair_encoders.append(RTNoEdgeInit(**module_args))
        
        module_args['function_type'] = args.function_type
        
        for i in range(args.num_layers):
            if i == 0:
                self.hyper_encoders.append(HRT(**module_args))
            else:
                self.hyper_encoders.append(HRTNoEdgeInit(**module_args))
        
        for i in range(args.sample_k):
            self.add_module("head_%d" % i, Decoder(args))

        # CFI: wrap heads in cross-modal future interaction decoder when requested
        if getattr(args, 'cfi', False):
            from .cfi import MARTCFIDecoder
            heads = [self._modules[f"head_{i}"] for i in range(args.sample_k)]
            self.cfi_decoder = MARTCFIDecoder(args, heads)
        else:
            self.cfi_decoder = None

    def forward(self, x_abs, x_rel, extra_feats=None, mu=None, sigma=None):
        inputs = []
        batch_size, num_agents, length, _ = x_abs.shape
        cur_pos = x_abs[:, :, [-1]].view(batch_size*num_agents, 1, -1).contiguous()

        if 'pos_x' in self.args.inputs and 'pos_y' in self.args.inputs:
            inputs.append(x_abs)
        if 'vel_x' in self.args.inputs and 'vel_y' in self.args.inputs:
            inputs.append(x_rel)

        inputs = torch.cat(inputs, dim=-1)
        if extra_feats is not None:
            inputs = torch.cat([inputs, extra_feats], dim=-1)
        inputs = inputs.view(batch_size*num_agents, length, -1).contiguous()
        
        inputs_fc = self.input_fc(inputs).view(batch_size*num_agents, length, self.args.model_dim)
        inputs_pos = self.pos_encoder(inputs_fc, num_a=batch_size*num_agents)
        inputs_pos = inputs_pos.view(batch_size, num_agents, length, self.args.model_dim)
        n_initial = self.input_fc2(inputs_pos.contiguous().view(batch_size, num_agents, length*self.args.model_dim))
        
        n_pair, e_pair = n_initial, None
        n_group, e_group, G = n_initial, None, None
        
        for i in range(self.args.num_layers):
            n_pair, e_pair = self.pair_encoders[i](n_pair, e_pair, return_edge=True)
            n_group, e_group, G = self.hyper_encoders[i](n_group, e_group, G, return_edge=True)
        
        n_final = torch.cat([n_initial, n_pair, n_group], dim=-1)

        if self.cfi_decoder is not None:
            # Returns (loc1, loc2) — caller handles dual loss and uses loc2 for val
            return self.cfi_decoder(n_final, cur_pos, mu, sigma)

        out_list = []
        for i in range(self.args.sample_k):
            out = self._modules["head_%d" % i](n_final, cur_pos)
            out_list.append(out[:, None, :, :])

        out = torch.cat(out_list, dim=2)
        out = out.view(batch_size, num_agents, self.args.sample_k, self.args.future_length, -1)

        return out
    
