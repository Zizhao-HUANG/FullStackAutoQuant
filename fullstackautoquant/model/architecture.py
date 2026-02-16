# 1. Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper: Causal 1D Conv with explicit left padding
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
    def forward(self, x):
        # x: (B, C, L)
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)

class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k1=5, d1=1, k2=5, d2=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=k1, dilation=d1)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=k2, dilation=d2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res_proj = None
        if in_channels != out_channels:
            self.res_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        h = self.norm(x)
        h = h.transpose(1, 2)  # (B, C, L)
        y = self.conv1(h)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.act(y)
        y = self.drop(y)
        res = h if self.res_proj is None else self.res_proj(h)
        out = (y + res).transpose(1, 2)  # back to (B, L, C_out)
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=96, nhead=6, head_dim=16, attn_dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = head_dim
        self.inner_dim = nhead * head_dim
        self.qkv = nn.Linear(d_model, 3 * self.inner_dim, bias=True)
        self.out = nn.Linear(self.inner_dim, d_model, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(attn_dropout)

    def forward(self, x, causal_mask: bool = True):
        # x: (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B, T, 3*H*Hd)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # reshape to heads
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # (B, H, T, Hd)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        # scaled dot-product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B,H,T,T)
        if causal_mask:
            # mask future: allow j <= i
            i = torch.arange(T, device=x.device)
            mask = i.unsqueeze(0) >= i.unsqueeze(1)  # (T,T) lower triangular
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # (B,H,T,Hd)
        out = out.transpose(1, 2).contiguous().view(B, T, self.nhead * self.head_dim)
        out = self.out(out)
        out = self.proj_drop(out)
        return out

class LocalSelfAttentionOverlap(nn.Module):
    def __init__(self, d_model=96, window_size=16, stride=8, nhead=6, head_dim=16, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model=d_model, nhead=nhead, head_dim=head_dim, attn_dropout=attn_dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, d_model),
        )
        self.drop2 = nn.Dropout(dropout)
        self.post_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        w, s = self.window_size, self.stride
        assert L >= w, "Sequence too short for windowed attention"
        # Number of windows covering [0, L)
        num_win = 1 + (max(L - w, 0) // s)
        # Prepare accumulators
        out = x.new_zeros(B, L, D)
        cnt = x.new_zeros(B, L, 1)
        for i in range(num_win):
            start = i * s
            end = start + w
            if end > L:
                # stop if window exceeds length (shouldn't due to num_win def, but safe)
                break
            seg = x[:, start:end, :]  # (B, w, D)
            h = self.ln1(seg)
            h_attn = self.mha(h, causal_mask=True)
            h = seg + self.drop1(h_attn)
            h2 = self.ln2(h)
            h_ffn = self.ffn(h2)
            h = h + self.drop2(h_ffn)
            h = self.post_ln(h)
            out[:, start:end, :] += h
            cnt[:, start:end, :] += 1
        # average overlapping contributions
        out = out / torch.clamp_min(cnt, 1.0)
        return out

# 2. Define the model class, inheriting from nn.Module
class Net(nn.Module):
    # REMINDER: Your __init__ must accept num_features and num_timesteps.
    def __init__(self, num_features: int = 6, num_timesteps: int = 72):
        """
        TCN_LocalAttn_GRU_L72_RankMSE_Balanced_v2
        ModelType: TimeSeries
        Fixed architecture and training hyperparameters embedded in attributes.
        """
        super().__init__()
        # Fixed Model Hyperparameters (static)
        self.model_name = "TCN_LocalAttn_GRU_L72_RankMSE_Balanced_v2"
        self.model_type = "TimeSeries"
        self.expected_timesteps = 72
        self.d_model = 96
        self.hidden_size = 48
        self.nhead = 6
        self.head_dim = 16
        self.dropout = 0.1
        self.window_size = 16
        self.window_stride = 8

        # Record training hyperparameters (for reference only; training loop not included)
        self.training_hyperparameters = {
            'n_epochs': 95,
            'lr': 2e-4,
            'early_stop': 10,
            'batch_size': 256,
            'weight_decay': 1e-4,
            'precision': 'bf16',
            'loss_fn': '0.7*PairwiseHinge(m=0.015)+0.3*MSE + TVR(4e-4)',
            'optimizer': 'AdamW',
            'optimizer_kwargs': {'betas': (0.9, 0.999), 'eps': 1e-08},
            'lr_scheduler': 'CosineAnnealingLRWithWarmup',
            'lr_scheduler_kwargs': {'T_max': 95, 'eta_min': 1e-06, 'warmup_steps': 1000},
            'seed': 42,
            'step_len': 72,
            'num_timesteps': 72,
            'window_length': 72,
            'window_stride': 8,
            'gradient_clip_norm': 0.8,
            'rank_loss_margin': 0.015,
            'rank_mse_blend_alpha': 0.7,
            'turnover_regularization_lambda': 4e-4,
        }

        # 1) Linear Embedding 6->96
        self.input_proj = nn.Linear(num_features, self.d_model)

        # 2) TCN Stem: three residual causal blocks
        # Block1: in 96 -> out 64 with residual 1x1 conv
        self.tcn_block1 = ResidualTCNBlock(
            in_channels=self.d_model, out_channels=64,
            k1=5, d1=1, k2=5, d2=2, dropout=self.dropout
        )
        # Block2: 64 -> 64 identity residual
        self.tcn_block2 = ResidualTCNBlock(
            in_channels=64, out_channels=64,
            k1=5, d1=2, k2=5, d2=4, dropout=self.dropout
        )
        # Block3: 64 -> 64 identity residual
        self.tcn_block3 = ResidualTCNBlock(
            in_channels=64, out_channels=64,
            k1=5, d1=4, k2=5, d2=4, dropout=self.dropout
        )
        # Post-TCN Projection: 64 -> 96
        self.post_tcn_proj = nn.Linear(64, self.d_model)

        # 3) Overlapping Local Self-Attention encoder
        self.local_attn = LocalSelfAttentionOverlap(
            d_model=self.d_model,
            window_size=self.window_size,
            stride=self.window_stride,
            nhead=self.nhead,
            head_dim=self.head_dim,
            attn_dropout=self.dropout,
            dropout=self.dropout,
        )

        # 4) GRU Head
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

        # 5) Output MLP
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: TimeSeries input of shape (N, S, F)
        Returns:
            prediction tensor of shape (N, 1)
        """
        # Ensure shape (B, L, F)
        assert x.dim() == 3, "Input must be 3D (batch, seq_len, features)"
        B, L, Fdim = x.shape
        # 1) Linear embedding
        z = self.input_proj(x)  # (B, L, 96)
        # 2) TCN Stem
        h = self.tcn_block1(z)
        h = self.tcn_block2(h)
        h = self.tcn_block3(h)
        h = self.post_tcn_proj(h)  # (B, L, 96)
        # 3) Overlapping Local Self-Attention with causal masking
        y = self.local_attn(h)  # (B, L, 96)
        # 4) GRU: take last hidden state
        out, h_n = self.gru(y)
        last_hidden = out[:, -1, :]  # (B, 48)
        # 5) Output MLP
        pred = self.head(last_hidden)  # (B, 1)
        return pred

# 3. Set the 'model_cls' variable to your defined class
model_cls = Net
