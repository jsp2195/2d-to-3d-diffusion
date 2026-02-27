import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


torch.manual_seed(42)
np.random.seed(42)


class WSConv3d(nn.Conv3d):
    def forward(self, x):
        w = self.weight
        w = w - w.mean(dim=(1, 2, 3, 4), keepdim=True)
        w = w / (w.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-5)
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = WSConv3d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = WSConv3d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_ch * 2)
        self.skip = WSConv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        gamma_beta = self.time_mlp(t_emb)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(h)
        h = h * (1 + gamma) + beta
        h = self.conv2(self.act2(h))
        return h + self.skip(x)


class SelfAttention3D(nn.Module):
    def __init__(self, channels: int, heads: int = 8, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv1d(channels, inner * 3, kernel_size=1)
        self.to_out = nn.Conv1d(inner, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        n = d * h * w
        h_in = x
        x = self.norm(x).view(b, c, n)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.heads, self.dim_head, n).permute(0, 1, 3, 2)
        k = k.view(b, self.heads, self.dim_head, n)
        v = v.view(b, self.heads, self.dim_head, n).permute(0, 1, 3, 2)

        scale = self.dim_head ** -0.5
        attn = torch.matmul(q, k) * scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, self.heads * self.dim_head, n)
        out = self.to_out(out).view(b, c, d, h, w)
        return out + h_in


class CrossAttention3D(nn.Module):
    def __init__(self, channels: int, context_dim: int, heads: int = 8, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Conv1d(channels, inner, kernel_size=1)
        self.to_k = nn.Linear(context_dim, inner)
        self.to_v = nn.Linear(context_dim, inner)
        self.to_out = nn.Conv1d(inner, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        n = d * h * w
        x_in = x
        x = self.norm(x).view(b, c, n)
        q = self.to_q(x).view(b, self.heads, self.dim_head, n).permute(0, 1, 3, 2)

        k = self.to_k(context).view(b, -1, self.heads, self.dim_head).permute(0, 2, 3, 1)
        v = self.to_v(context).view(b, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)

        scale = self.dim_head ** -0.5
        attn = torch.matmul(q, k) * scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, self.heads * self.dim_head, n)
        out = self.to_out(out).view(b, c, d, h, w)
        return out + x_in


class CondEncoder2DTokens(nn.Module):
    def __init__(self, in_channels: int, token_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, token_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, token_dim),
            nn.SiLU(),
        )

    def forward(self, cond2d: torch.Tensor) -> torch.Tensor:
        # cond2d: [B, K, H, W]
        feat = self.net(cond2d)
        # feat: [B, Cctx, H/8, W/8]
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        # tokens: [B, Ntokens, Cctx]
        return tokens


class DownStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, with_attn: bool):
        super().__init__()
        self.res1 = ResBlock3D(in_ch, out_ch, time_dim)
        self.res2 = ResBlock3D(out_ch, out_ch, time_dim)
        self.attn = SelfAttention3D(out_ch) if with_attn else nn.Identity()
        self.down = WSConv3d(out_ch, out_ch, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int, with_attn: bool):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.res1 = ResBlock3D(out_ch + skip_ch, out_ch, time_dim)
        self.res2 = ResBlock3D(out_ch, out_ch, time_dim)
        self.attn = SelfAttention3D(out_ch) if with_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        return x


class ConditionalUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 8,
        k_slices: int = 5,
        base_dim: int = 64,
        levels: int = 4,
        groupnorm_groups: int = 8,
        context_dim: int = 256,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        if levels != 4:
            raise ValueError("levels must be 4 for channel multipliers (1,2,4,8)")

        self.use_checkpoint = use_checkpoint
        self.in_channels = in_channels
        self.model_out_channels = out_channels

        time_dim = base_dim * 4
        chs = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]

        self.time_emb = SinusoidalTimeEmbedding(base_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.cond_encoder = CondEncoder2DTokens(k_slices, token_dim=context_dim)

        self.in_conv = WSConv3d(in_channels, chs[0], kernel_size=3, padding=1)

        self.down1 = DownStage(chs[0], chs[0], time_dim, with_attn=False)   # 128->64
        self.down2 = DownStage(chs[0], chs[1], time_dim, with_attn=False)   # 64->32
        self.down3 = DownStage(chs[1], chs[2], time_dim, with_attn=True)    # 32->16
        self.down4 = DownStage(chs[2], chs[3], time_dim, with_attn=True)    # 16->8

        self.mid_res1 = ResBlock3D(chs[3], chs[3], time_dim, groups=groupnorm_groups)
        self.mid_self = SelfAttention3D(chs[3])
        self.mid_cross = CrossAttention3D(chs[3], context_dim)
        self.mid_res2 = ResBlock3D(chs[3], chs[3], time_dim, groups=groupnorm_groups)

        self.up4 = UpStage(chs[3], chs[3], chs[2], time_dim, with_attn=True)   # 8->16
        self.up3 = UpStage(chs[2], chs[2], chs[1], time_dim, with_attn=True)   # 16->32
        self.up2 = UpStage(chs[1], chs[1], chs[0], time_dim, with_attn=False)  # 32->64
        self.up1 = UpStage(chs[0], chs[0], chs[0], time_dim, with_attn=False)  # 64->128

        self.out_norm = nn.GroupNorm(groupnorm_groups, chs[0])
        self.out_act = nn.SiLU()
        self.out_conv = WSConv3d(chs[0], out_channels, kernel_size=3, padding=1)

    def _ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond2d: torch.Tensor) -> torch.Tensor:
        # x_t: [B, C, D, H, W]
        # t: [B]
        # cond2d: [B, K, H, W]
        t_emb = self.time_mlp(self.time_emb(t))
        context = self.cond_encoder(cond2d)

        x = self.in_conv(x_t)
        x, s1 = self._ckpt(self.down1, x, t_emb)
        x, s2 = self._ckpt(self.down2, x, t_emb)
        x, s3 = self._ckpt(self.down3, x, t_emb)
        x, s4 = self._ckpt(self.down4, x, t_emb)

        x = self._ckpt(self.mid_res1, x, t_emb)
        x = self._ckpt(self.mid_self, x)
        x = self._ckpt(self.mid_cross, x, context)
        x = self._ckpt(self.mid_res2, x, t_emb)

        x = self._ckpt(self.up4, x, s4, t_emb)
        x = self._ckpt(self.up3, x, s3, t_emb)
        x = self._ckpt(self.up2, x, s2, t_emb)
        x = self._ckpt(self.up1, x, s1, t_emb)

        out = self.out_conv(self.out_act(self.out_norm(x)))
        # out: [B, 2*C, D, H, W] (pred + variance)
        return out
