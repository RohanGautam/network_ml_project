import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import os

from .basemodel import MLP

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=1000):
        super(PositionalEncoding, self).__init__()
        positional_encodings = torch.zeros(max_seq_len, d_model)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term  = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        positional_encodings[:, 0::2] = torch.sin(positions * div_term)
        positional_encodings[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer('positional_encodings', positional_encodings.unsqueeze(0))

    def forward(self, x):
        return x + self.positional_encodings[:, :x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.pe        = PositionalEncoding(d_model)
        self.query     = nn.Linear(d_model, d_model)
        self.key       = nn.Linear(d_model, d_model)
        self.value     = nn.Linear(d_model, d_model)
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        queries    = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys       = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            mask        = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-1e8'))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        values     = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        context    = torch.matmul(attn_probs, values)
        context    = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_projection(context)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in  = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound   = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in  = m.embed_dim
            fan_out = m.embed_dim
            bound   = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


class Decoder(nn.Module):

    def __init__(self, args) -> None:
        super(Decoder, self).__init__()
        min_scale: float   = 1e-3
        self.args          = args
        self.input_size    = self.args.hidden_size
        self.hidden_size   = self.args.hidden_size
        self.future_steps  = args.pred_length
        self.num_modes     = 20
        self.min_scale     = min_scale

        self.lstm  = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.lstm2 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.self_attention = MultiHeadSelfAttention(self.hidden_size, 4)

        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.multihead_proj_global = nn.Sequential(
            nn.Linear(self.input_size, self.num_modes * self.hidden_size),
            nn.LayerNorm(self.num_modes * self.hidden_size),
            nn.ReLU(inplace=True))

        self.apply(init_weights)

    def forward(self, global_embed, hidden_state, cn, shift_values, max_values, batch_split):
        dev          = global_embed.device
        shift_values = shift_values.squeeze(0)
        N            = hidden_state.size(0)

        global_embed   = self.multihead_proj_global(global_embed).view(12, -1, self.num_modes, self.hidden_size)
        global_embed   = global_embed.transpose(1, 2)   # [H, F, N, D]
        local_embed    = hidden_state.repeat(self.num_modes, 1, 1)
        cn             = cn.repeat(self.num_modes, 1, 1)

        global_embed_1 = global_embed.reshape(self.future_steps, -1, self.hidden_size)
        hn_1           = local_embed.reshape(-1, self.hidden_size)
        cn_1           = cn.reshape(-1, self.hidden_size)

        output1_list = []
        for t in range(self.future_steps):
            hn_1, cn_1 = self.lstm(global_embed_1[t], (hn_1, cn_1))
            output1_list.append(hn_1)
        output1   = torch.stack(output1_list)
        output1_t = output1.transpose(0, 1)
        loc1      = self.loc(output1_t).view(self.num_modes, -1, self.future_steps, 2)
        scale1    = F.elu_(self.scale(output1_t), alpha=1.0) + 1.0 + self.min_scale
        scale1    = scale1.view(self.num_modes, -1, self.future_steps, 2)

        # future social interaction
        loc1_          = loc1 * max_values.unsqueeze(-2) + shift_values.unsqueeze(-2)
        loc1_          = loc1_.transpose(1, 2).reshape(self.num_modes * self.future_steps, -1, 2).transpose(-1, -2)
        distance_whole = torch.abs(loc1_.unsqueeze(-1) - loc1_.unsqueeze(-2))
        modal_mask     = torch.ones(self.num_modes, self.num_modes, device=dev)
        social_modal_future = torch.zeros(self.future_steps, self.num_modes, N, self.hidden_size, device=dev)

        output1_tmp = output1.detach().reshape(self.future_steps, self.num_modes, -1, self.hidden_size)
        for left, right in batch_split:
            now_n            = right - left
            distance_b       = distance_whole[:, :, left:right, left:right]
            social_mask_b    = (distance_b[:, 0] < 10) & (distance_b[:, 1] < 10)
            social_mask_b    = social_mask_b.reshape(self.num_modes, self.future_steps, now_n, now_n)
            social_mask_full = social_mask_b.any(0)
            joint_mask_b     = social_mask_full.repeat(1, 20, 20).bool()
            output1_tmp_b    = output1_tmp[:, :, left:right, :].reshape(self.future_steps, -1, self.hidden_size)
            social_out_b     = self.self_attention(output1_tmp_b, joint_mask_b)
            social_out_b     = social_out_b.reshape(self.future_steps, self.num_modes, now_n, self.hidden_size)
            social_modal_future[:, :, left:right, :] = social_out_b

        social_modal_future = social_modal_future.reshape(self.future_steps, -1, self.hidden_size)

        global_embed_2 = global_embed_1
        output2_list   = []
        hn_2           = local_embed.reshape(-1, self.hidden_size)
        cn_2           = cn.reshape(-1, self.hidden_size)
        for t in range(self.future_steps):
            cn_2 = cn_2 + social_modal_future[t]
            hn_2 = hn_2 + torch.tanh(cn_2)
            hn_2, cn_2 = self.lstm2(global_embed_2[t], (hn_2, cn_2))
            output2_list.append(hn_2)
        output2   = torch.stack(output2_list)
        output2_t = output2.transpose(0, 1)
        loc2      = self.loc(output2_t).view(self.num_modes, -1, self.future_steps, 2)
        scale2    = F.elu_(self.scale(output2_t), alpha=1.0) + 1.0 + self.min_scale
        scale2    = scale2.view(self.num_modes, -1, self.future_steps, 2)

        return (loc1, scale1), (loc2, scale2)
