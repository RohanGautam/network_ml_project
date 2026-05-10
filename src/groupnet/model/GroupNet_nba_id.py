"""GroupNet variant that injects per-agent entity-type embedding at the input.

Differences vs. GroupNet_nba.GroupNet:
    - adds nn.Embedding(num_entity_types, embed_dim)
    - encoder input is concat([x, y, vx, vy, embed]) -> in_dim = 4 + embed_dim
    - forward()/inference() expect data['agent_ids']: [B, N] long tensor

The dataset is responsible for placing agents in the canonical order
[TeamA(5), TeamB(5), Ball(1)] so the encoders' built-in positional `add_category`
remains valid alongside the learned embedding.
"""

import torch
from torch import nn

from model.GroupNet_nba import (
    PastEncoder, FutureEncoder, Decoder, Normal,
)
from model.utils import initialize_weights


class GroupNetWithID(nn.Module):
    def __init__(self, args, device, embed_dim=4, num_entity_types=3):
        super().__init__()
        self.device = device
        self.args = args
        self.embed_dim = embed_dim

        in_dim = 4 + embed_dim
        scale_num = 2 + len(args.hyper_scales)

        self.entity_embed = nn.Embedding(num_entity_types, embed_dim)
        nn.init.normal_(self.entity_embed.weight, mean=0.0, std=0.1)

        self.past_encoder = PastEncoder(args, in_dim=in_dim)
        self.future_encoder = FutureEncoder(args, in_dim=in_dim)
        self.pz_layer = nn.Linear(scale_num * args.hidden_dim, 2 * args.zdim)
        if args.learn_prior:
            initialize_weights(self.pz_layer.modules())
        self.decoder = Decoder(args)
        self.param_annealers = nn.ModuleList()

    def set_device(self, device):
        self.device = device
        self.to(device)

    def _build_inputs(self, traj, vel, agent_ids, B, N, T):
        """Concat (traj, vel, broadcasted entity embedding) along feature dim."""
        embed = self.entity_embed(agent_ids)                               # [B, N, d]
        embed = embed.unsqueeze(2).expand(B, N, T, self.embed_dim)         # [B, N, T, d]
        embed = embed.reshape(B * N, T, self.embed_dim).to(traj.dtype)
        return torch.cat([traj, vel, embed], dim=-1)                       # [B*N, T, 4+d]

    @staticmethod
    def _loss_l2(pred, target, batch_size):
        return (target - pred).pow(2).sum() / batch_size / pred.shape[1]

    @staticmethod
    def _loss_kl(qz, pz, batch_size, agent_num, min_clip):
        loss = qz.kl(pz).sum() / (batch_size * agent_num)
        return loss.clamp_min_(min_clip)

    @staticmethod
    def _loss_diverse(pred, target, batch_size):
        diff = target.unsqueeze(1) - pred
        avg_dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
        return avg_dist.min(dim=1)[0].mean()

    def _make_pz(self, past_feature, sample_num=1):
        """Build the prior distribution; matches the parent's logic."""
        device = past_feature.device
        if sample_num > 1:
            past_feature = past_feature.repeat_interleave(sample_num, dim=0)
        if self.args.learn_prior:
            return past_feature, Normal(params=self.pz_layer(past_feature))
        return past_feature, Normal(
            mu=torch.zeros(past_feature.shape[0], self.args.zdim, device=device),
            logvar=torch.zeros(past_feature.shape[0], self.args.zdim, device=device),
        )

    def forward(self, data):
        device = self.device
        B = data['past_traj'].shape[0]
        N = data['past_traj'].shape[1]
        T_p = self.args.past_length
        T_f = self.args.future_length

        if self.args.ztype != 'gaussian':
            raise ValueError(f"unsupported ztype: {self.args.ztype}")

        past_traj = data['past_traj'].view(B * N, T_p, 2).to(device).contiguous()
        future_traj = data['future_traj'].view(B * N, T_f, 2).to(device).contiguous()
        agent_ids = data['agent_ids'].to(device)

        past_vel = past_traj[:, 1:] - past_traj[:, :-1, :]
        past_vel = torch.cat([past_vel[:, [0]], past_vel], dim=1)
        future_vel = future_traj - torch.cat([past_traj[:, [-1]], future_traj[:, :-1, :]], dim=1)
        cur_location = past_traj[:, [-1]]

        inputs_past = self._build_inputs(past_traj, past_vel, agent_ids, B, N, T_p)
        inputs_post = self._build_inputs(future_traj, future_vel, agent_ids, B, N, T_f)

        past_feature = self.past_encoder(inputs_past, B, N)
        qz_param = self.future_encoder(inputs_post, B, N, past_feature)
        qz = Normal(params=qz_param)
        qz_sampled = qz.rsample()

        _, pz = self._make_pz(past_feature, sample_num=1)

        pred_traj, recover_traj = self.decoder(
            past_feature, qz_sampled, B, N, past_traj, cur_location, sample_num=1,
        )
        loss_pred = self._loss_l2(pred_traj, future_traj, B)
        loss_recover = self._loss_l2(recover_traj, past_traj, B)
        loss_kl = self._loss_kl(qz, pz, B, N, self.args.min_clip)

        K = 20
        past_feature_rep, pz_div = self._make_pz(past_feature, sample_num=K)
        pz_sampled = pz_div.rsample()
        diverse_pred, _ = self.decoder(
            past_feature_rep, pz_sampled, B, N, past_traj, cur_location,
            sample_num=K, mode='inference',
        )
        loss_diverse = self._loss_diverse(diverse_pred, future_traj, B)

        total_loss = loss_pred + loss_recover + loss_kl + loss_diverse
        return total_loss, loss_pred.item(), loss_recover.item(), loss_kl.item(), loss_diverse.item()

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()

    def inference(self, data):
        device = self.device
        B = data['past_traj'].shape[0]
        N = data['past_traj'].shape[1]
        T_p = self.args.past_length

        past_traj = data['past_traj'].view(B * N, T_p, 2).to(device).contiguous()
        agent_ids = data['agent_ids'].to(device)

        past_vel = past_traj[:, 1:] - past_traj[:, :-1, :]
        past_vel = torch.cat([past_vel[:, [0]], past_vel], dim=1)
        cur_location = past_traj[:, [-1]]

        inputs_past = self._build_inputs(past_traj, past_vel, agent_ids, B, N, T_p)
        past_feature = self.past_encoder(inputs_past, B, N)

        K = self.args.sample_k
        past_feature_rep, pz = self._make_pz(past_feature, sample_num=K)
        z = pz.rsample()
        diverse_pred, _ = self.decoder(
            past_feature_rep, z, B, N, past_traj, cur_location,
            sample_num=K, mode='inference',
        )
        return diverse_pred.permute(1, 0, 2, 3)
