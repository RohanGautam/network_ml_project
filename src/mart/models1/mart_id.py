"""MART variant that injects a learned entity-type embedding at the input.

Mirrors GroupNetWithID's mechanism: at each timestep, every agent's standard
input feature [pos_x, pos_y, vel_x, vel_y] is augmented with a learned
`embed_dim`-vector indexed by the agent's entity type (TeamA=0, TeamB=1,
Ball=2). Everything else (RT pair encoders, HRT hyper encoders, decoder heads)
is unchanged from MART.

Why this is needed for the NBA setup:
    MART has no positional agent encoding (PositionalAgentEncoding only encodes
    time), so the relational transformer body is permutation-equivariant over
    agents. The dataloader places agents in canonical [TeamA(5), TeamB(5),
    Ball(1)] order, but MART can't read that order without a per-token type
    signal. The embedding breaks that symmetry.

Only `input_fc` is resized to accept the extra `embed_dim` features; all other
weights are bit-identical to MART (same shapes, same init).
"""

import torch
from torch import nn

from .mart import MART


NUM_ENTITY_TYPES = 3  # TeamA, TeamB, Ball


class MART_ID(MART):
    def __init__(self, args, embed_dim=None, num_entity_types=NUM_ENTITY_TYPES):
        super().__init__(args)
        self.embed_dim = int(embed_dim if embed_dim is not None else args.embed_dim)
        self.num_entity_types = num_entity_types

        self.entity_embedding = nn.Embedding(num_entity_types, self.embed_dim)
        # Resize the first FC to accept the extra embed_dim channels. All other
        # downstream shapes (model_dim, input_fc2, transformers, decoder) are
        # unchanged so MART's recipe carries over verbatim.
        self.input_fc = nn.Linear(self.input_dim + self.embed_dim, args.model_dim)

    def forward(self, x_abs, x_rel, agent_ids):
        """
        Args:
            x_abs:     [B, N, T_p, 2]
            x_rel:     [B, N, T_p, 2]
            agent_ids: [B, N] long
        Returns:
            out: [B, N, K, T_f, 2]
        """
        batch_size, num_agents, length, _ = x_abs.shape
        cur_pos = x_abs[:, :, [-1]].view(batch_size * num_agents, 1, -1).contiguous()

        # ---- Build the standard MART input feature ----
        inputs = []
        if 'pos_x' in self.args.inputs and 'pos_y' in self.args.inputs:
            inputs.append(x_abs)
        if 'vel_x' in self.args.inputs and 'vel_y' in self.args.inputs:
            inputs.append(x_rel)
        inputs = torch.cat(inputs, dim=-1)   # [B, N, T_p, input_dim]

        # ---- Inject entity-type embedding (the only addition vs MART) ----
        embed = self.entity_embedding(agent_ids)                          # [B, N, embed_dim]
        embed = embed.unsqueeze(2).expand(
            batch_size, num_agents, length, self.embed_dim,
        ).to(inputs.dtype)                                                # [B, N, T_p, embed_dim]
        inputs = torch.cat([inputs, embed], dim=-1)                       # [B, N, T_p, input_dim+embed_dim]

        inputs = inputs.view(batch_size * num_agents, length, -1).contiguous()

        # ---- Rest is byte-for-byte MART.forward ----
        inputs_fc = self.input_fc(inputs).view(
            batch_size * num_agents, length, self.args.model_dim,
        )
        inputs_pos = self.pos_encoder(inputs_fc, num_a=batch_size * num_agents)
        inputs_pos = inputs_pos.view(
            batch_size, num_agents, length, self.args.model_dim,
        )
        n_initial = self.input_fc2(
            inputs_pos.contiguous().view(
                batch_size, num_agents, length * self.args.model_dim,
            )
        )

        n_pair, e_pair = n_initial, None
        n_group, e_group, G = n_initial, None, None

        for i in range(self.args.num_layers):
            n_pair, e_pair = self.pair_encoders[i](n_pair, e_pair, return_edge=True)
            n_group, e_group, G = self.hyper_encoders[i](n_group, e_group, G, return_edge=True)

        n_final = torch.cat([n_initial, n_pair, n_group], dim=-1)

        out_list = []
        for i in range(self.args.sample_k):
            out = self._modules["head_%d" % i](n_final, cur_pos)
            out_list.append(out[:, None, :, :])

        out = torch.cat(out_list, dim=2)
        out = out.view(
            batch_size, num_agents, self.args.sample_k, self.args.future_length, -1,
        )
        return out
