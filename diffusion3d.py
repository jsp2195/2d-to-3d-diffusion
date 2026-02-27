import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext


torch.manual_seed(42)
np.random.seed(42)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).float()


class GaussianDiffusion3D(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        channels: int,
        timesteps: int = 1000,
        objective: str = "v",
        p2_gamma: float = 0.0,
        p2_k: float = 1.0,
        vlb_weight: float = 1e-3,
        cfg_drop_prob: float = 0.1,
    ):
        super().__init__()
        if objective not in {"eps", "v"}:
            raise ValueError("objective must be 'eps' or 'v'")

        self.model = model
        self.channels = channels
        self.timesteps = timesteps
        self.objective = objective
        self.vlb_weight = vlb_weight
        self.cfg_drop_prob = cfg_drop_prob

        beta = cosine_beta_schedule(timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alpha_bar", torch.sqrt(1.0 / alpha_bar))
        self.register_buffer("sqrt_recipm1_alpha_bar", torch.sqrt(1.0 / alpha_bar - 1.0))

        posterior_variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_variance = posterior_variance.clamp(min=1e-20)
        posterior_log_variance_clipped = torch.log(posterior_variance)
        posterior_mean_coef1 = beta * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_mean_coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(alpha) / (1.0 - alpha_bar)

        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

        p2 = (p2_k + alpha_bar / (1 - alpha_bar)) ** -p2_gamma
        self.register_buffer("p2_loss_weight", p2)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        xt = a * x0 + b * noise
        return xt, noise

    def _predict_x0_from_eps(self, xt, t, eps):
        return self.sqrt_recip_alpha_bar[t].view(-1, 1, 1, 1, 1) * xt - self.sqrt_recipm1_alpha_bar[t].view(-1, 1, 1, 1, 1) * eps

    def _predict_eps_x0(self, xt: torch.Tensor, t: torch.Tensor, pred_main: torch.Tensor):
        a = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        if self.objective == "eps":
            eps = pred_main
            x0 = self._predict_x0_from_eps(xt, t, eps)
        else:
            v = pred_main
            x0 = a * xt - b * v
            eps = a * v + b * xt
        return eps, x0

    def _true_posterior(self, x0, xt, t):
        mean = self.posterior_mean_coef1[t].view(-1, 1, 1, 1, 1) * x0 + self.posterior_mean_coef2[t].view(-1, 1, 1, 1, 1) * xt
        log_var = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1, 1)
        var = torch.exp(log_var)
        return mean, var, log_var

    def _model_log_variance(self, t: torch.Tensor, pred_var: torch.Tensor):
        min_log = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1, 1)
        max_log = torch.log(self.beta[t]).view(-1, 1, 1, 1, 1)
        frac = (torch.tanh(pred_var) + 1.0) * 0.5
        model_log_var = frac * max_log + (1.0 - frac) * min_log
        model_var = torch.exp(model_log_var)
        return model_var, model_log_var

    def model_predictions(self, xt: torch.Tensor, t: torch.Tensor, cond2d: torch.Tensor, guidance_scale: float = 1.0):
        out_cond = self.model(xt, t, cond2d)
        pred_cond, pred_var = torch.split(out_cond, self.channels, dim=1)

        if guidance_scale != 1.0:
            null_cond = torch.zeros_like(cond2d)
            out_null = self.model(xt, t, null_cond)
            pred_null, _ = torch.split(out_null, self.channels, dim=1)
            pred = pred_null + guidance_scale * (pred_cond - pred_null)
        else:
            pred = pred_cond

        eps, x0 = self._predict_eps_x0(xt, t, pred)
        x0 = x0.clamp(-5.0, 5.0)
        return eps, x0, pred_var

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor, cond2d: torch.Tensor):
        b = x0.shape[0]
        drop_mask = (torch.rand(b, device=x0.device) < self.cfg_drop_prob).float().view(-1, 1, 1, 1)
        cond_used = cond2d * (1.0 - drop_mask)

        xt, noise = self.q_sample(x0, t)
        out = self.model(xt, t, cond_used)
        pred_main, pred_var = torch.split(out, self.channels, dim=1)

        target = noise if self.objective == "eps" else self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1) * noise - self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1) * x0

        loss_simple = F.mse_loss(pred_main, target, reduction="none").mean(dim=(1, 2, 3, 4))
        loss_simple = (loss_simple * self.p2_loss_weight[t]).mean()

        _, x0_pred = self._predict_eps_x0(xt, t, pred_main)
        true_mean, true_var, true_log_var = self._true_posterior(x0, xt, t)
        model_var, model_log_var = self._model_log_variance(t, pred_var)
        model_mean = self.posterior_mean_coef1[t].view(-1, 1, 1, 1, 1) * x0_pred + self.posterior_mean_coef2[t].view(-1, 1, 1, 1, 1) * xt

        kl = 0.5 * (
            true_log_var - model_log_var + (model_var + (model_mean - true_mean) ** 2) / true_var - 1.0
        )
        kl = kl.mean(dim=(1, 2, 3, 4))

        nll = 0.5 * (
            np.log(2.0 * np.pi) + model_log_var + (x0 - model_mean) ** 2 / model_var
        )
        nll = nll.mean(dim=(1, 2, 3, 4))

        decoder_nll = torch.where(t == 0, nll, kl)
        loss_vlb = decoder_nll.mean()

        return loss_simple + self.vlb_weight * loss_vlb, loss_simple.detach(), loss_vlb.detach()

    @torch.no_grad()
    def p_sample(self, xt: torch.Tensor, t_int: int, cond2d: torch.Tensor, guidance_scale: float = 3.0, amp: bool = True):
        b = xt.shape[0]
        t = torch.full((b,), t_int, device=xt.device, dtype=torch.long)
        context = torch.cuda.amp.autocast(enabled=(amp and xt.is_cuda)) if hasattr(torch.cuda, "amp") else nullcontext()
        with context:
            eps, x0, pred_var = self.model_predictions(xt, t, cond2d, guidance_scale=guidance_scale)
            model_var, model_log_var = self._model_log_variance(t, pred_var)
            model_mean = self.posterior_mean_coef1[t].view(-1, 1, 1, 1, 1) * x0 + self.posterior_mean_coef2[t].view(-1, 1, 1, 1, 1) * xt

        if t_int == 0:
            return model_mean
        noise = torch.randn_like(xt)
        return model_mean + torch.exp(0.5 * model_log_var) * noise

    @torch.no_grad()
    def sample(self, cond2d: torch.Tensor, shape, guidance_scale: float = 3.0, amp: bool = True):
        x = torch.randn(shape, device=cond2d.device)
        for t in range(self.timesteps - 1, -1, -1):
            x = self.p_sample(x, t, cond2d, guidance_scale=guidance_scale, amp=amp)
        return x

    @torch.no_grad()
    def sample_ddim(self, cond2d: torch.Tensor, shape, sampling_steps: int = 50, guidance_scale: float = 3.0, eta: float = 0.0, amp: bool = True):
        device = cond2d.device
        x = torch.randn(shape, device=device)

        times = torch.linspace(self.timesteps - 1, 0, sampling_steps, device=device).long()
        next_times = torch.cat([times[1:], torch.tensor([-1], device=device, dtype=torch.long)])

        for t, t_next in zip(times, next_times):
            b = x.shape[0]
            t_batch = torch.full((b,), int(t.item()), device=device, dtype=torch.long)
            context = torch.cuda.amp.autocast(enabled=(amp and x.is_cuda)) if hasattr(torch.cuda, "amp") else nullcontext()
            with context:
                eps, x0, _ = self.model_predictions(x, t_batch, cond2d, guidance_scale=guidance_scale)

            if t_next < 0:
                x = x0
                continue

            ab_t = self.alpha_bar[t]
            ab_next = self.alpha_bar[t_next]
            sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t) * (1 - ab_t / ab_next))
            c = torch.sqrt(torch.clamp(1 - ab_next - sigma ** 2, min=0.0))
            noise = torch.randn_like(x)
            x = torch.sqrt(ab_next) * x0 + c * eps + sigma * noise

        return x

    @torch.no_grad()
    def estimate_x0(self, x0: torch.Tensor, t: torch.Tensor, cond2d: torch.Tensor):
        xt, _ = self.q_sample(x0, t)
        eps, x0_pred, _ = self.model_predictions(xt, t, cond2d, guidance_scale=1.0)
        return x0_pred
