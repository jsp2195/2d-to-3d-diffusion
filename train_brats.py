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


def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.999) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def save_montage(volume: torch.Tensor, save_path: str) -> None:
    # volume shape: [1, D, H, W]
    vol = volume.detach().cpu().numpy()[0]
    # vol shape: [D, H, W]
    z_list = [32, 64, 96]
    fig, axes = plt.subplots(1, len(z_list), figsize=(12, 4))
    for i, z in enumerate(z_list):
        axes[i].imshow(vol[z], cmap="gray")
        axes[i].set_title(f"z={z}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--k_slices", type=int, default=5)
    parser.add_argument("--base_dim", type=int, default=32)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_brats")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--objective", type=str, default="v", choices=["eps", "v"])
    parser.add_argument("--sample_every", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = BraTSDataset(args.root_dir, split="train", k_slices=args.k_slices, seed=42)
    val_ds = BraTSDataset(args.root_dir, split="val", k_slices=args.k_slices, seed=42)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ConditionalUNet3D(
        in_channels=1,
        out_channels=1,
        k_slices=args.k_slices,
        base_dim=args.base_dim,
        levels=args.levels,
        groupnorm_groups=8,
        depth_size=128,
    ).to(device)

    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    diffusion = GaussianDiffusion3D(
        model=model,
        timesteps=args.timesteps,
        beta_start=1e-4,
        beta_end=0.02,
        objective=args.objective,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for cond2d, target3d in train_loader:
            # cond2d shape: [B, K, H, W]
            # target3d shape: [B, 1, D, H, W]
            cond2d = cond2d.to(device, non_blocking=True)
            target3d = target3d.to(device, non_blocking=True)

            b = target3d.shape[0]
            t = torch.randint(0, args.timesteps, (b,), device=device, dtype=torch.long)
            # t shape: [B]

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                loss = diffusion.p_losses(target3d, t, cond2d)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            update_ema(ema_model, model, decay=args.ema_decay)
            train_losses.append(loss.item())

        mean_train = float(np.mean(train_losses)) if train_losses else 0.0

        model.eval()
        val_losses = []
        with torch.no_grad():
            for cond2d, target3d in val_loader:
                # cond2d shape: [B, K, H, W]
                # target3d shape: [B, 1, D, H, W]
                cond2d = cond2d.to(device, non_blocking=True)
                target3d = target3d.to(device, non_blocking=True)
                b = target3d.shape[0]
                t = torch.randint(0, args.timesteps, (b,), device=device, dtype=torch.long)
                # t shape: [B]
                with autocast(enabled=torch.cuda.is_available()):
                    val_loss = diffusion.p_losses(target3d, t, cond2d)
                val_losses.append(val_loss.item())

        mean_val = float(np.mean(val_losses)) if val_losses else 0.0
        print(f"Epoch {epoch}/{args.epochs} | train_loss={mean_train:.6f} | val_loss={mean_val:.6f}")

        ckpt = {
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "config": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))

        if epoch % args.sample_every == 0 and len(val_ds) > 0:
            cond_ex, _ = val_ds[0]
            # cond_ex shape: [K, H, W]
            cond_ex = cond_ex.unsqueeze(0).to(device)
            # cond_ex shape: [1, K, H, W]
            sample = diffusion.sample(ema_model, cond_ex, shape=(1, 1, 128, 128, 128))
            # sample shape: [1, 1, D, H, W]
            montage_path = os.path.join(args.save_dir, f"sample_epoch_{epoch:04d}.png")
            save_montage(sample[0], montage_path)


if __name__ == "__main__":
    main()
