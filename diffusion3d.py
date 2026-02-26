import torch
import torch.nn as nn
import torch.nn.functional as F


class Diffusion3D(nn.Module):
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bar[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alpha", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar),
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # t shape: [B]
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        return t

    def forward_diffusion(self, x_0: torch.Tensor, t: torch.Tensor):
        # x_0 shape: [B, 1, D, H, W]
        # t shape: [B]
        noise = torch.randn_like(x_0)
        # noise shape: [B, 1, D, H, W]
        a = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1)
        # a shape: [B, 1, 1, 1, 1]
        b = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        # b shape: [B, 1, 1, 1, 1]
        x_t = a * x_0 + b * noise
        # x_t shape: [B, 1, D, H, W]
        return x_t, noise

    def loss(self, model: nn.Module, x_0: torch.Tensor, cond2d: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        # x_0 shape: [B, 1, D, H, W]
        # cond2d shape: [B, 1, H, W]
        if t is None:
            t = self.sample_timesteps(x_0.shape[0], x_0.device)
            # t shape: [B]
        x_t, noise = self.forward_diffusion(x_0, t)
        # x_t shape: [B, 1, D, H, W]
        # noise shape: [B, 1, D, H, W]
        pred_noise = model(x_t, t, cond2d)
        # pred_noise shape: [B, 1, D, H, W]
        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, model: nn.Module, cond2d: torch.Tensor, shape: tuple, device: torch.device):
        # cond2d shape: [B, 1, H, W]
        # shape tuple: (B, 1, D, H, W)
        b, c, d, h, w = shape
        x = torch.randn(b, c, d, h, w, device=device)
        # x shape: [B, 1, D, H, W]

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            # t shape: [B]
            pred_noise = model(x, t, cond2d)
            # pred_noise shape: [B, 1, D, H, W]

            beta_t = self.betas[t].view(-1, 1, 1, 1, 1)
            # beta_t shape: [B, 1, 1, 1, 1]
            alpha_t = self.alphas[t].view(-1, 1, 1, 1, 1)
            # alpha_t shape: [B, 1, 1, 1, 1]
            sqrt_one_minus_ab_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
            # sqrt_one_minus_ab_t shape: [B, 1, 1, 1, 1]
            sqrt_recip_alpha_t = self.sqrt_recip_alpha[t].view(-1, 1, 1, 1, 1)
            # sqrt_recip_alpha_t shape: [B, 1, 1, 1, 1]

            model_mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_ab_t) * pred_noise)
            # model_mean shape: [B, 1, D, H, W]

            if i > 0:
                noise = torch.randn_like(x)
                # noise shape: [B, 1, D, H, W]
                posterior_var_t = self.posterior_variance[t].view(-1, 1, 1, 1, 1)
                # posterior_var_t shape: [B, 1, 1, 1, 1]
                x = model_mean + torch.sqrt(posterior_var_t) * noise
                # x shape: [B, 1, D, H, W]
            else:
                x = model_mean
                # x shape: [B, 1, D, H, W]

        return x


if __name__ == "__main__":
    from unet3d_conditional import UNet3DConditional

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3DConditional().to(device)
    diffusion = Diffusion3D(device=device).to(device)

    x0 = torch.randn(1, 1, 128, 128, 128, device=device)
    cond2d = torch.randn(1, 1, 128, 128, device=device)
    loss = diffusion.loss(model, x0, cond2d)
    print(float(loss))
