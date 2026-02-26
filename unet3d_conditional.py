import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)
np.random.seed(42)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: [B]
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        # freqs shape: [half]
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        # args shape: [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        # emb shape: [B, dim]
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x shape: [B, Cin, D, H, W]
        # t_emb shape: [B, time_dim]
        h = self.conv1(self.act1(self.norm1(x)))
        # h shape: [B, Cout, D, H, W]
        t_add = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # t_add shape: [B, Cout, 1, 1, 1]
        h = h + t_add
        h = self.conv2(self.act2(self.norm2(h)))
        # h shape: [B, Cout, D, H, W]
        return h + self.skip(x)


class ConditionEncoder2D(nn.Module):
    def __init__(self, in_ch: int, out_channels: Tuple[int, int, int, int], groups: int = 8):
        super().__init__()
        c0, c1, c2, c3 = out_channels
        self.block0 = nn.Sequential(
            nn.Conv2d(in_ch, c0, kernel_size=3, padding=1),
            nn.GroupNorm(groups, c0),
            nn.SiLU(),
            nn.Conv2d(c0, c0, kernel_size=3, padding=1),
            nn.GroupNorm(groups, c0),
            nn.SiLU(),
        )
        self.down1 = nn.Conv2d(c0, c1, kernel_size=4, stride=2, padding=1)
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.GroupNorm(groups, c1),
            nn.SiLU(),
        )
        self.down2 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.GroupNorm(groups, c2),
            nn.SiLU(),
        )
        self.down3 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)
        self.block3 = nn.Sequential(
            nn.GroupNorm(groups, c3),
            nn.SiLU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.GroupNorm(groups, c3),
            nn.SiLU(),
        )

    def forward(self, cond2d: torch.Tensor) -> List[torch.Tensor]:
        # cond2d shape: [B, K, H, W]
        f0 = self.block0(cond2d)
        # f0 shape: [B, C0, 128, 128]
        f1 = self.block1(self.down1(f0))
        # f1 shape: [B, C1, 64, 64]
        f2 = self.block2(self.down2(f1))
        # f2 shape: [B, C2, 32, 32]
        f3 = self.block3(self.down3(f2))
        # f3 shape: [B, C3, 16, 16]
        return [f0, f1, f2, f3]


class ConditionalUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        k_slices: int = 5,
        base_dim: int = 32,
        levels: int = 3,
        groupnorm_groups: int = 8,
        depth_size: int = 128,
    ):
        super().__init__()
        if levels != 3:
            raise ValueError("This implementation requires levels=3")

        self.depth_size = depth_size
        self.base_dim = base_dim

        ch0 = base_dim
        ch1 = base_dim * 2
        ch2 = base_dim * 4
        time_dim = base_dim * 4

        self.time_embed = SinusoidalTimeEmbedding(base_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.cond_encoder = ConditionEncoder2D(k_slices, (ch0, ch1, ch2, ch2), groups=groupnorm_groups)

        self.stem = nn.Conv3d(in_channels, ch0, kernel_size=3, padding=1)
        self.cond_fuse_stem = nn.Conv3d(ch0 + ch0, ch0, kernel_size=1)

        self.down0 = ResBlock3D(ch0, ch0, time_dim, groups=groupnorm_groups)
        self.cond_fuse_d0 = nn.Conv3d(ch0 + ch0, ch0, kernel_size=1)
        self.downsample0 = nn.Conv3d(ch0, ch1, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))

        self.down1 = ResBlock3D(ch1, ch1, time_dim, groups=groupnorm_groups)
        self.cond_fuse_d1 = nn.Conv3d(ch1 + ch1, ch1, kernel_size=1)
        self.downsample1 = nn.Conv3d(ch1, ch2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))

        self.down2 = ResBlock3D(ch2, ch2, time_dim, groups=groupnorm_groups)
        self.cond_fuse_d2 = nn.Conv3d(ch2 + ch2, ch2, kernel_size=1)
        self.downsample2 = nn.Conv3d(ch2, ch2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))

        self.mid = ResBlock3D(ch2, ch2, time_dim, groups=groupnorm_groups)
        self.cond_fuse_mid = nn.Conv3d(ch2 + ch2, ch2, kernel_size=1)

        self.upsample2 = nn.ConvTranspose3d(ch2, ch2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.up2 = ResBlock3D(ch2 + ch2, ch2, time_dim, groups=groupnorm_groups)
        self.cond_fuse_u2 = nn.Conv3d(ch2 + ch2, ch2, kernel_size=1)

        self.upsample1 = nn.ConvTranspose3d(ch2, ch1, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.up1 = ResBlock3D(ch1 + ch1, ch1, time_dim, groups=groupnorm_groups)
        self.cond_fuse_u1 = nn.Conv3d(ch1 + ch1, ch1, kernel_size=1)

        self.upsample0 = nn.ConvTranspose3d(ch1, ch0, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.up0 = ResBlock3D(ch0 + ch0, ch0, time_dim, groups=groupnorm_groups)
        self.cond_fuse_u0 = nn.Conv3d(ch0 + ch0, ch0, kernel_size=1)

        self.out_norm = nn.GroupNorm(groupnorm_groups, ch0)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv3d(ch0, out_channels, kernel_size=3, padding=1)

        self.depth_embed = nn.Embedding(depth_size, base_dim)
        self.depth_proj_stem = nn.Conv3d(base_dim, ch0, kernel_size=1)
        self.depth_proj_d0 = nn.Conv3d(base_dim, ch0, kernel_size=1)
        self.depth_proj_d1 = nn.Conv3d(base_dim, ch1, kernel_size=1)
        self.depth_proj_d2 = nn.Conv3d(base_dim, ch2, kernel_size=1)
        self.depth_proj_mid = nn.Conv3d(base_dim, ch2, kernel_size=1)
        self.depth_proj_u2 = nn.Conv3d(base_dim, ch2, kernel_size=1)
        self.depth_proj_u1 = nn.Conv3d(base_dim, ch1, kernel_size=1)
        self.depth_proj_u0 = nn.Conv3d(base_dim, ch0, kernel_size=1)

    def _depth_feature(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        # x shape: [B, C, D, H, W]
        b, _, d, h, w = x.shape
        depth_ids = torch.arange(d, device=x.device)
        # depth_ids shape: [D]
        depth_emb = self.depth_embed(depth_ids)
        # depth_emb shape: [D, base_dim]
        depth_emb = depth_emb.transpose(0, 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # depth_emb shape: [1, base_dim, D, 1, 1]
        depth_emb = depth_emb.expand(b, -1, -1, h, w)
        # depth_emb shape: [B, base_dim, D, H, W]
        return proj(depth_emb)

    def _lift_cond(self, cond2d_feat: torch.Tensor, depth: int) -> torch.Tensor:
        # cond2d_feat shape: [B, Cc, H, W]
        lifted = cond2d_feat.unsqueeze(2).repeat(1, 1, depth, 1, 1)
        # lifted shape: [B, Cc, D, H, W]
        return lifted

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond2d: torch.Tensor) -> torch.Tensor:
        # x_t shape: [B, 1, D, H, W]
        # t shape: [B]
        # cond2d shape: [B, K, H, W]
        cond_feats = self.cond_encoder(cond2d)
        # cond_feats[0] shape: [B, C0, 128, 128]
        # cond_feats[1] shape: [B, C1, 64, 64]
        # cond_feats[2] shape: [B, C2, 32, 32]
        # cond_feats[3] shape: [B, C2, 16, 16]

        t_emb = self.time_mlp(self.time_embed(t))
        # t_emb shape: [B, time_dim]

        x = self.stem(x_t)
        # x shape: [B, C0, D, 128, 128]
        cond0_3d = self._lift_cond(cond_feats[0], x.shape[2])
        # cond0_3d shape: [B, C0, D, 128, 128]
        x = self.cond_fuse_stem(torch.cat([x, cond0_3d], dim=1))
        # x shape: [B, C0, D, 128, 128]
        x = x + self._depth_feature(x, self.depth_proj_stem)
        # x shape: [B, C0, D, 128, 128]

        s0 = self.down0(x, t_emb)
        # s0 shape: [B, C0, D, 128, 128]
        s0 = self.cond_fuse_d0(torch.cat([s0, cond0_3d], dim=1))
        # s0 shape: [B, C0, D, 128, 128]
        s0 = s0 + self._depth_feature(s0, self.depth_proj_d0)
        # s0 shape: [B, C0, D, 128, 128]

        x1 = self.downsample0(s0)
        # x1 shape: [B, C1, D, 64, 64]

        c1_3d = self._lift_cond(cond_feats[1], x1.shape[2])
        # c1_3d shape: [B, C1, D, 64, 64]
        s1 = self.down1(x1, t_emb)
        # s1 shape: [B, C1, D, 64, 64]
        s1 = self.cond_fuse_d1(torch.cat([s1, c1_3d], dim=1))
        # s1 shape: [B, C1, D, 64, 64]
        s1 = s1 + self._depth_feature(s1, self.depth_proj_d1)
        # s1 shape: [B, C1, D, 64, 64]

        x2 = self.downsample1(s1)
        # x2 shape: [B, C2, D, 32, 32]

        c2_3d = self._lift_cond(cond_feats[2], x2.shape[2])
        # c2_3d shape: [B, C2, D, 32, 32]
        s2 = self.down2(x2, t_emb)
        # s2 shape: [B, C2, D, 32, 32]
        s2 = self.cond_fuse_d2(torch.cat([s2, c2_3d], dim=1))
        # s2 shape: [B, C2, D, 32, 32]
        s2 = s2 + self._depth_feature(s2, self.depth_proj_d2)
        # s2 shape: [B, C2, D, 32, 32]

        x3 = self.downsample2(s2)
        # x3 shape: [B, C2, D, 16, 16]

        c3_3d = self._lift_cond(cond_feats[3], x3.shape[2])
        # c3_3d shape: [B, C2, D, 16, 16]
        x3 = self.mid(x3, t_emb)
        # x3 shape: [B, C2, D, 16, 16]
        x3 = self.cond_fuse_mid(torch.cat([x3, c3_3d], dim=1))
        # x3 shape: [B, C2, D, 16, 16]
        x3 = x3 + self._depth_feature(x3, self.depth_proj_mid)
        # x3 shape: [B, C2, D, 16, 16]

        u2 = self.upsample2(x3)
        # u2 shape: [B, C2, D, 32, 32]
        u2 = torch.cat([u2, s2], dim=1)
        # u2 shape: [B, 2*C2, D, 32, 32]
        u2 = self.up2(u2, t_emb)
        # u2 shape: [B, C2, D, 32, 32]
        u2 = self.cond_fuse_u2(torch.cat([u2, c2_3d], dim=1))
        # u2 shape: [B, C2, D, 32, 32]
        u2 = u2 + self._depth_feature(u2, self.depth_proj_u2)
        # u2 shape: [B, C2, D, 32, 32]

        u1 = self.upsample1(u2)
        # u1 shape: [B, C1, D, 64, 64]
        u1 = torch.cat([u1, s1], dim=1)
        # u1 shape: [B, 2*C1, D, 64, 64]
        u1 = self.up1(u1, t_emb)
        # u1 shape: [B, C1, D, 64, 64]
        u1 = self.cond_fuse_u1(torch.cat([u1, c1_3d], dim=1))
        # u1 shape: [B, C1, D, 64, 64]
        u1 = u1 + self._depth_feature(u1, self.depth_proj_u1)
        # u1 shape: [B, C1, D, 64, 64]

        u0 = self.upsample0(u1)
        # u0 shape: [B, C0, D, 128, 128]
        u0 = torch.cat([u0, s0], dim=1)
        # u0 shape: [B, 2*C0, D, 128, 128]
        u0 = self.up0(u0, t_emb)
        # u0 shape: [B, C0, D, 128, 128]
        u0 = self.cond_fuse_u0(torch.cat([u0, cond0_3d], dim=1))
        # u0 shape: [B, C0, D, 128, 128]
        u0 = u0 + self._depth_feature(u0, self.depth_proj_u0)
        # u0 shape: [B, C0, D, 128, 128]

        out = self.out_conv(self.out_act(self.out_norm(u0)))
        # out shape: [B, 1, D, 128, 128]
        return out
