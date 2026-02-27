import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset_brats import BraTSDataset
from diffusion3d import GaussianDiffusion3D
from unet3d_conditional import ConditionalUNet3D


torch.manual_seed(42)
np.random.seed(42)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--k_slices", type=int, default=5)
    p.add_argument("--base_dim", type=int, default=64)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--save_dir", type=str, default="./checkpoints_brats")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_warmup_steps", type=int, default=2000)
    p.add_argument("--objective", type=str, default="v", choices=["eps", "v"])
    p.add_argument("--sample_every", type=int, default=1)
    p.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--sampling_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--cfg_drop_prob", type=float, default=0.1)
    p.add_argument("--vlb_weight", type=float, default=1e-3)
    p.add_argument("--modalities", nargs="+", default=["t1", "t2", "flair", "t1ce"])
    p.add_argument("--cond_modality", type=str, default="t1")
    p.add_argument("--patch_overlap", action="store_true")
    p.add_argument("--overlap_stride", type=int, default=64)
    p.add_argument("--single_modality_mode", action="store_true")
    p.add_argument("--use_checkpoint", action="store_true")
    return p.parse_args()


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.num_updates = 0
        self.shadow = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    def update(self, model: torch.nn.Module, warmup_steps: int = 2000):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        if self.num_updates < warmup_steps:
            d = 0.0
        with torch.no_grad():
            idx = 0
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                self.shadow[idx].mul_(d).add_(p.detach(), alpha=1.0 - d)
                idx += 1

    def copy_to(self, model: torch.nn.Module):
        with torch.no_grad():
            idx = 0
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                p.copy_(self.shadow[idx])
                idx += 1

    def state_dict(self):
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]
        self.shadow = [t.clone() for t in state["shadow"]]


def mse_3d(x, y):
    return torch.mean((x - y) ** 2)


def ssim_3d(x, y, c1=0.01 ** 2, c2=0.03 ** 2):
    # x,y: [B, C, D, H, W]
    mu_x = x.mean(dim=(2, 3, 4), keepdim=True)
    mu_y = y.mean(dim=(2, 3, 4), keepdim=True)
    sigma_x = ((x - mu_x) ** 2).mean(dim=(2, 3, 4), keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=(2, 3, 4), keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(2, 3, 4), keepdim=True)
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return (num / den).mean()


def save_montage(volume: torch.Tensor, save_path: str):
    # volume: [C, D, H, W]
    v = volume[0].detach().cpu().numpy()
    z_list = [32, 64, 96]
    fig, axes = plt.subplots(1, len(z_list), figsize=(12, 4))
    for i, z in enumerate(z_list):
        axes[i].imshow(v[z], cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"z={z}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.single_modality_mode:
        modalities = [args.modalities[0]]
    else:
        modalities = args.modalities

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = BraTSDataset(
        args.root_dir,
        split="train",
        k_slices=args.k_slices,
        seed=42,
        modalities=modalities,
        cond_modality=args.cond_modality,
        patch_overlap=args.patch_overlap,
        overlap_stride=args.overlap_stride,
    )
    val_ds = BraTSDataset(
        args.root_dir,
        split="val",
        k_slices=args.k_slices,
        seed=42,
        modalities=modalities,
        cond_modality=args.cond_modality,
        patch_overlap=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    in_ch = len(modalities)
    model = ConditionalUNet3D(
        in_channels=in_ch,
        out_channels=in_ch * 2,
        k_slices=args.k_slices,
        base_dim=args.base_dim,
        levels=args.levels,
        use_checkpoint=args.use_checkpoint,
    ).to(device)

    ema_model = copy.deepcopy(model).to(device)
    ema = EMA(model, decay=args.ema_decay)

    diffusion = GaussianDiffusion3D(
        model=model,
        channels=in_ch,
        timesteps=args.timesteps,
        objective=args.objective,
        vlb_weight=args.vlb_weight,
        cfg_drop_prob=args.cfg_drop_prob,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        optimizer.zero_grad(set_to_none=True)
        for i, (cond2d, target3d) in enumerate(train_loader):
            cond2d = cond2d.to(device, non_blocking=True)
            target3d = target3d.to(device, non_blocking=True)

            b = target3d.shape[0]
            t = torch.randint(0, args.timesteps, (b,), device=device, dtype=torch.long)

            with autocast(enabled=torch.cuda.is_available()):
                loss, loss_simple, loss_vlb = diffusion.p_losses(target3d, t, cond2d)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model, warmup_steps=args.ema_warmup_steps)
                global_step += 1

            train_losses.append((loss_simple.item(), loss_vlb.item()))

        ema.copy_to(ema_model)

        model.eval()
        val_loss_list, val_mse_list, val_ssim_list = [], [], []
        with torch.no_grad():
            for cond2d, target3d in val_loader:
                cond2d = cond2d.to(device, non_blocking=True)
                target3d = target3d.to(device, non_blocking=True)
                t = torch.randint(0, args.timesteps, (target3d.shape[0],), device=device, dtype=torch.long)

                with autocast(enabled=torch.cuda.is_available()):
                    loss, _, _ = diffusion.p_losses(target3d, t, cond2d)
                    x0_pred = diffusion.estimate_x0(target3d, t, cond2d)
                    m = mse_3d(x0_pred, target3d)
                    s = ssim_3d(x0_pred, target3d)

                val_loss_list.append(loss.item())
                val_mse_list.append(m.item())
                val_ssim_list.append(s.item())

        train_simple = np.mean([x[0] for x in train_losses]) if train_losses else 0.0
        train_vlb = np.mean([x[1] for x in train_losses]) if train_losses else 0.0
        val_loss = float(np.mean(val_loss_list)) if val_loss_list else 0.0
        val_mse = float(np.mean(val_mse_list)) if val_mse_list else 0.0
        val_ssim = float(np.mean(val_ssim_list)) if val_ssim_list else 0.0

        print(
            f"Epoch {epoch}/{args.epochs} | train_simple={train_simple:.6f} | train_vlb={train_vlb:.6f} | "
            f"val_loss={val_loss:.6f} | val_mse3d={val_mse:.6f} | val_ssim3d={val_ssim:.6f}"
        )

        ckpt = {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": vars(args),
            "modalities": modalities,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))

        if epoch % args.sample_every == 0 and len(val_ds) > 0:
            cond_ex, _ = val_ds[0]
            cond_ex = cond_ex.unsqueeze(0).to(device)
            with autocast(enabled=torch.cuda.is_available()):
                if args.sampler == "ddim":
                    sample = diffusion.sample_ddim(
                        cond_ex,
                        shape=(1, in_ch, 128, 128, 128),
                        sampling_steps=args.sampling_steps,
                        guidance_scale=args.guidance_scale,
                    )
                else:
                    sample = diffusion.sample(
                        cond_ex,
                        shape=(1, in_ch, 128, 128, 128),
                        guidance_scale=args.guidance_scale,
                    )
            save_montage(sample[0], os.path.join(args.save_dir, f"sample_epoch_{epoch:04d}.png"))


if __name__ == "__main__":
    main()
