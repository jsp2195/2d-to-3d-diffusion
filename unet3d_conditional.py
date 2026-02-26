import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: [B]
        half = self.dim // 2
        emb_scale = torch.log(torch.tensor(10000.0, device=t.device)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        # emb shape: [B, dim/2]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # emb shape: [B, dim]
        return emb


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C_in, D, H, W]
        # t_emb shape: [B, time_dim]
        h = self.conv1(x)
        # h shape: [B, C_out, D, H, W]
        h = self.gn1(h)
        h = self.act(h)

        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # t shape: [B, C_out, 1, 1, 1]
        h = h + t
        # h shape: [B, C_out, D, H, W]

        h = self.conv2(h)
        # h shape: [B, C_out, D, H, W]
        h = self.gn2(h)
        h = self.act(h)

        out = h + self.skip(x)
        # out shape: [B, C_out, D, H, W]
        return out


class DownBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.res1 = ResBlock3D(in_ch, out_ch, time_dim)
        self.res2 = ResBlock3D(out_ch, out_ch, time_dim)
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        # x shape: [B, C_in, D, H, W]
        x = self.res1(x, t_emb)
        # x shape: [B, C_out, D, H, W]
        x = self.res2(x, t_emb)
        # x shape: [B, C_out, D, H, W]
        skip = x
        # skip shape: [B, C_out, D, H, W]
        x = self.down(x)
        # x shape: [B, C_out, D/2, H/2, W/2]
        return x, skip


class UpBlock3D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.res1 = ResBlock3D(out_ch + skip_ch, out_ch, time_dim)
        self.res2 = ResBlock3D(out_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C_in, D, H, W]
        # skip shape: [B, C_skip, 2D, 2H, 2W]
        x = self.up(x)
        # x shape: [B, C_out, 2D, 2H, 2W]
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            # x shape: [B, C_out, D_skip, H_skip, W_skip]
        x = torch.cat([x, skip], dim=1)
        # x shape: [B, C_out + C_skip, D_skip, H_skip, W_skip]
        x = self.res1(x, t_emb)
        # x shape: [B, C_out, D_skip, H_skip, W_skip]
        x = self.res2(x, t_emb)
        # x shape: [B, C_out, D_skip, H_skip, W_skip]
        return x


class UNet3DConditional(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_dim: int = 32, depth_levels: int = 3, max_depth: int = 128):
        super().__init__()
        if depth_levels != 3:
            raise ValueError("depth_levels must be 3")
        self.base_dim = base_dim
        self.depth_levels = depth_levels

        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.depth_embed = nn.Embedding(max_depth, base_dim)

        self.in_conv = nn.Conv3d(in_channels, base_dim, kernel_size=3, padding=1)

        self.down1 = DownBlock3D(base_dim, base_dim * 2, time_dim)
        self.down2 = DownBlock3D(base_dim * 2, base_dim * 4, time_dim)
        self.down3 = DownBlock3D(base_dim * 4, base_dim * 8, time_dim)

        self.mid1 = ResBlock3D(base_dim * 8, base_dim * 8, time_dim)
        self.mid2 = ResBlock3D(base_dim * 8, base_dim * 8, time_dim)

        self.up3 = UpBlock3D(base_dim * 8, base_dim * 8, base_dim * 4, time_dim)
        self.up2 = UpBlock3D(base_dim * 4, base_dim * 4, base_dim * 2, time_dim)
        self.up1 = UpBlock3D(base_dim * 2, base_dim * 2, base_dim, time_dim)

        self.out_block = ResBlock3D(base_dim, base_dim, time_dim)
        self.out_conv = nn.Conv3d(base_dim, out_channels, kernel_size=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond2d: torch.Tensor) -> torch.Tensor:
        # x_t shape: [B, 1, D, H, W]
        # t shape: [B]
        # cond2d shape: [B, 1, H, W]
        b, _, d, h, w = x_t.shape

        cond3d = cond2d.unsqueeze(2).repeat(1, 1, d, 1, 1)
        # cond3d shape: [B, 1, D, H, W]

        x = torch.cat([x_t, cond3d], dim=1)
        # x shape: [B, 2, D, H, W]

        x = self.in_conv(x)
        # x shape: [B, base_dim, D, H, W]

        depth_ids = torch.arange(d, device=x_t.device)
        # depth_ids shape: [D]
        depth_emb = self.depth_embed(depth_ids)
        # depth_emb shape: [D, base_dim]
        depth_emb = depth_emb.permute(1, 0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # depth_emb shape: [1, base_dim, D, 1, 1]
        depth_emb = depth_emb.expand(b, -1, -1, h, w)
        # depth_emb shape: [B, base_dim, D, H, W]
        x = x + depth_emb
        # x shape: [B, base_dim, D, H, W]

        t_emb = self.time_mlp(t)
        # t_emb shape: [B, base_dim*4]

        x, s1 = self.down1(x, t_emb)
        # x shape: [B, base_dim*2, D/2, H/2, W/2]
        # s1 shape: [B, base_dim*2, D, H, W]
        x, s2 = self.down2(x, t_emb)
        # x shape: [B, base_dim*4, D/4, H/4, W/4]
        # s2 shape: [B, base_dim*4, D/2, H/2, W/2]
        x, s3 = self.down3(x, t_emb)
        # x shape: [B, base_dim*8, D/8, H/8, W/8]
        # s3 shape: [B, base_dim*8, D/4, H/4, W/4]

        x = self.mid1(x, t_emb)
        # x shape: [B, base_dim*8, D/8, H/8, W/8]
        x = self.mid2(x, t_emb)
        # x shape: [B, base_dim*8, D/8, H/8, W/8]

        x = self.up3(x, s3, t_emb)
        # x shape: [B, base_dim*4, D/4, H/4, W/4]
        x = self.up2(x, s2, t_emb)
        # x shape: [B, base_dim*2, D/2, H/2, W/2]
        x = self.up1(x, s1, t_emb)
        # x shape: [B, base_dim, D, H, W]

        x = self.out_block(x, t_emb)
        # x shape: [B, base_dim, D, H, W]
        x = self.out_conv(x)
        # x shape: [B, 1, D, H, W]
        return x


if __name__ == "__main__":
    model = UNet3DConditional()
    x_t = torch.randn(1, 1, 128, 128, 128)
    t = torch.randint(0, 1000, (1,))
    cond2d = torch.randn(1, 1, 128, 128)
    y = model(x_t, t, cond2d)
    print(y.shape)
