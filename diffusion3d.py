import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)
np.random.seed(42)


class GaussianDiffusion3D(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        objective: str = "v",
    ):
        super().__init__()
        if objective not in {"eps", "v"}:
            raise ValueError("objective must be 'eps' or 'v'")

        self.model = model
        self.timesteps = timesteps
        self.objective = objective

        beta = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alpha_bar[:-1]], dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

        posterior_variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_variance[0] = 1e-20
        posterior_mean_coef1 = beta * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_mean_coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(alpha) / (1.0 - alpha_bar)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        # x_0 shape: [B, 1, D, H, W]
        # t shape: [B]
        if noise is None:
            noise = torch.randn_like(x_0)
            # noise shape: [B, 1, D, H, W]
        a = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1)
        # a shape: [B, 1, 1, 1, 1]
        b = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        # b shape: [B, 1, 1, 1, 1]
        x_t = a * x_0 + b * noise
        # x_t shape: [B, 1, D, H, W]
        return x_t, noise

    def _target_from(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.objective == "eps":
            # target shape: [B, 1, D, H, W]
            return noise
        a = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        v = a * noise - b * x_0
        # v shape: [B, 1, D, H, W]
        return v

    def p_losses(self, x_0: torch.Tensor, t: torch.Tensor, cond2d: torch.Tensor) -> torch.Tensor:
        # x_0 shape: [B, 1, D, H, W]
        # cond2d shape: [B, K, H, W]
        x_t, noise = self.q_sample(x_0, t)
        # x_t shape: [B, 1, D, H, W]
        pred = self.model(x_t, t, cond2d)
        # pred shape: [B, 1, D, H, W]
        target = self._target_from(x_0, noise, t)
        # target shape: [B, 1, D, H, W]
        return F.mse_loss(pred, target)

    def _predict_eps_and_x0(self, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor):
        # x_t shape: [B, 1, D, H, W]
        # pred shape: [B, 1, D, H, W]
        a = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)

        if self.objective == "eps":
            eps = pred
            x0 = (x_t - b * eps) / a
        else:
            v = pred
            x0 = a * x_t - b * v
            eps = a * v + b * x_t
        # eps shape: [B, 1, D, H, W]
        # x0 shape: [B, 1, D, H, W]
        return eps, x0

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t_int: int, cond2d: torch.Tensor):
        # x_t shape: [B, 1, D, H, W]
        b = x_t.shape[0]
        t = torch.full((b,), t_int, device=x_t.device, dtype=torch.long)
        # t shape: [B]

        pred = model(x_t, t, cond2d)
        # pred shape: [B, 1, D, H, W]
        eps, x0 = self._predict_eps_and_x0(x_t, t, pred)
        x0 = x0.clamp(-5.0, 5.0)

        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1, 1, 1)
        # coef1 shape: [B, 1, 1, 1, 1]
        # coef2 shape: [B, 1, 1, 1, 1]
        mean = coef1 * x0 + coef2 * x_t
        # mean shape: [B, 1, D, H, W]

        if t_int == 0:
            return mean

        var = self.posterior_variance[t].view(-1, 1, 1, 1, 1)
        # var shape: [B, 1, 1, 1, 1]
        noise = torch.randn_like(x_t)
        # noise shape: [B, 1, D, H, W]
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, cond2d: torch.Tensor, shape):
        # cond2d shape: [B, K, H, W]
        # shape is tuple: (B, 1, D, H, W)
        x = torch.randn(shape, device=cond2d.device)
        # x shape: [B, 1, D, H, W]
        for t in range(self.timesteps - 1, -1, -1):
            x = self.p_sample(model, x, t, cond2d)
            # x shape: [B, 1, D, H, W]
        return x
