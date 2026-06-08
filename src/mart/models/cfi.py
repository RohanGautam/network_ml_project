"""MART-CFI: Cross-modal Future Interaction decoder for MART.

Adapts the CFI mechanism from HHT-CFI (laplace_decoder_joint.py) to MART's
one-shot MLP decoder architecture.

How it works:
  Branch 1: MART's existing K independent MLP heads → loc1 [B, N, K, T_f, 2]
  CFI:      Denormalize loc1 to feet → encode positions → self-attention across
            all K×N mode-agent tokens per future timestep → pool to [B, N, K, D]
  Branch 2: Second set of K MLP heads that receive n_final fused with CFI context
            → loc2 [B, N, K, T_f, 2]
  Loss:     compute_loss(loc1) + compute_loss(loc2)  (caller handles this)
  Inference: use loc2.mean(dim=K) — Branch 2 is the refined output.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(1024, 512)):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _Decoder(nn.Module):
    """Mirror of mart.py Decoder — duplicated here to avoid circular import."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        multiplier = len(args.hyper_scales) + 1
        self.mlp = _MLP(
            args.model_dim * multiplier,
            args.future_length * 2,
            hidden_dims=(args.decoder_hidden_dim, args.decoder_hidden_dim // 2),
        )

    def forward(self, final_feature, cur_location):
        out = self.mlp(final_feature).view(-1, self.args.future_length, 2)
        if not self.args.pred_rel:
            out = out + cur_location
        return out


class _MultiHeadSelfAttention(nn.Module):
    """Lightweight multi-head self-attention over a sequence of tokens."""

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [S, L, D]  S=batch of sequences, L=sequence length, D=d_model
        S, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q(x).reshape(S, L, H, Dh).transpose(1, 2)   # [S, H, L, Dh]
        k = self.k(x).reshape(S, L, H, Dh).transpose(1, 2)
        v = self.v(x).reshape(S, L, H, Dh).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [S, H, L, L]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                                # [S, H, L, Dh]
        out = out.transpose(1, 2).reshape(S, L, D)
        return self.out(out)                                       # [S, L, D]


class MARTCFIDecoder(nn.Module):
    """Drop-in replacement for MART's K independent decoder heads.

    Wraps the existing K Branch-1 heads unchanged, adds CFI cross-modal
    attention, and introduces K Branch-2 heads that use the CFI context.

    Args:
        args:          model config (same object passed to MART)
        branch1_heads: MART's existing _modules["head_k"] dict or ModuleList
    """

    def __init__(self, args, branch1_heads):
        super().__init__()
        self.args = args
        K = args.sample_k
        T_f = args.future_length
        # multiplier mirrors MART's Decoder: len(hyper_scales) + 1 = 3
        node_dim = args.model_dim * (len(args.hyper_scales) + 1)

        # Branch 1: borrow MART's existing heads (shared weights, not cloned)
        self.branch1 = nn.ModuleList(branch1_heads)

        # CFI attention over K×N tokens per future timestep
        # Project (x, y) trajectory positions → cfi_dim features per token
        cfi_dim = args.model_dim
        self.cfi_pos_proj = nn.Sequential(
            nn.Linear(T_f * 2, cfi_dim),
            nn.ReLU(inplace=True),
        )
        self.cfi_attn = _MultiHeadSelfAttention(cfi_dim, num_heads=4)
        self.cfi_norm = nn.LayerNorm(cfi_dim)

        # Fuse n_final (node_dim) + CFI context (cfi_dim) → node_dim for Branch 2
        self.context_fc = nn.Sequential(
            nn.Linear(node_dim + cfi_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(inplace=True),
        )

        # Branch 2: fresh K heads with same MLP architecture as Branch 1
        self.branch2 = nn.ModuleList(
            [_Decoder(args) for _ in range(K)]
        )

    def forward(self, n_final, cur_pos, mu, sigma):
        """
        Args:
            n_final:  [B, N, node_dim]   MART's fused node features
            cur_pos:  [B*N, 1, 2]        last observed position (z-scored)
            mu:       [2]                training-split mean (for denorm)
            sigma:    [2]                training-split std  (for denorm)

        Returns:
            loc1: [B, N, K, T_f, 2]   Branch 1 (raw MART predictions)
            loc2: [B, N, K, T_f, 2]   Branch 2 (CFI-refined predictions)
        """
        B, N, D = n_final.shape
        K = len(self.branch1)

        # ── Branch 1 ──────────────────────────────────────────────────────────
        b1_out = []
        for head in self.branch1:
            out = head(n_final, cur_pos)           # [B*N, T_f, 2]
            b1_out.append(out.view(B, N, self.args.future_length, 2))
        loc1 = torch.stack(b1_out, dim=2)          # [B, N, K, T_f, 2]

        # ── CFI: cross-modal attention ────────────────────────────────────────
        # Denormalize to feet so distance-based interactions are meaningful
        mu_d = mu.to(n_final.device).view(1, 1, 1, 1, 2)
        sigma_d = sigma.to(n_final.device).view(1, 1, 1, 1, 2)
        loc1_ft = loc1 * sigma_d + mu_d            # [B, N, K, T_f, 2], feet

        # Encode each agent-mode pair's full T_f trajectory as one D-dim token
        # [B, N, K, T_f*2] → [B, N, K, cfi_dim]
        traj_feats = self.cfi_pos_proj(
            loc1_ft.flatten(-2)                    # [B, N, K, T_f*2]
        )                                          # [B, N, K, cfi_dim]

        # Self-attention across all K×N tokens (flatten N and K into one sequence)
        # Treat B as the batch dim for attention: [B, K*N, cfi_dim]
        tokens = traj_feats.permute(0, 2, 1, 3).reshape(B, K * N, -1)
        cfi_ctx = self.cfi_attn(tokens)            # [B, K*N, cfi_dim]
        cfi_ctx = self.cfi_norm(tokens + cfi_ctx)  # residual + norm
        # Reshape back to [B, N, K, cfi_dim] and average over K to get per-agent summary
        cfi_ctx = cfi_ctx.view(B, K, N, -1).permute(0, 2, 1, 3)  # [B, N, K, cfi_dim]
        cfi_summary = cfi_ctx.mean(dim=2)          # [B, N, cfi_dim]

        # ── Branch 2 ──────────────────────────────────────────────────────────
        # Fuse n_final with the CFI summary → refined node features
        n_refined = self.context_fc(
            torch.cat([n_final, cfi_summary], dim=-1)   # [B, N, D+cfi_dim]
        )                                               # [B, N, D]

        b2_out = []
        for head in self.branch2:
            out = head(n_refined, cur_pos)         # [B*N, T_f, 2]
            b2_out.append(out.view(B, N, self.args.future_length, 2))
        loc2 = torch.stack(b2_out, dim=2)          # [B, N, K, T_f, 2]

        return loc1, loc2
