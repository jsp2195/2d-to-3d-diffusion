import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset_brats import BraTSDataset
from diffusion3d import GaussianDiffusion3D
from unet3d_conditional import ConditionalUNet3D


torch.manual_seed(42)
np.random.seed(42)
from scipy.ndimage import gaussian_filter

from mpl_toolkits.mplot3d import Axes3D

def save_3d_render(volume: torch.Tensor, save_path: str):
    vol = volume.detach().cpu().numpy()[0]

    # Choose threshold (top intensities only)
    threshold = np.percentile(vol, 90)

    coords = np.argwhere(vol > threshold)
    values = vol[vol > threshold]

    # Normalize grayscale
    values = (values - values.min()) / (values.max() - values.min() + 1e-8)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        coords[:, 2],  # X
        coords[:, 1],  # Y
        coords[:, 0],  # Z
        c=values,
        cmap="gray",
        s=1,
        alpha=0.6,
    )

    ax.set_xlim(0, vol.shape[2])
    ax.set_ylim(0, vol.shape[1])
    ax.set_zlim(0, vol.shape[0])
    ax.set_box_aspect(vol.shape)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.999) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
            

def save_montage(cond2d, volume, cond_indices, save_path):
    cond = cond2d.detach().cpu().numpy()
    vol = volume.detach().cpu().numpy()[0]
    cond_indices = cond_indices.cpu().numpy().tolist()

    d = vol.shape[0]

    # Global evenly spaced slices
    n_global = 8
    global_indices = np.linspace(0, d - 1, n_global).astype(int)

    n_cols = max(len(cond_indices), n_global)

    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9))

    # Clear all axes
    for r in range(3):
        for c in range(n_cols):
            axes[r, c].axis("off")

    # Row 1 — True conditioning slices
    start = (n_cols - len(cond_indices)) // 2
    for i, z in enumerate(cond_indices):
        axes[0, start + i].imshow(cond[i], cmap="gray")
        axes[0, start + i].set_title(f"Cond z={z}")

    # Row 2 — Generated reconstruction at those exact depths
    for i, z in enumerate(cond_indices):
        axes[1, start + i].imshow(vol[z], cmap="gray")
        axes[1, start + i].set_title(f"Recon z={z}")

    # Row 3 — Global volume
    for i, z in enumerate(global_indices):
        axes[2, i].imshow(vol[z], cmap="gray")
        axes[2, i].set_title(f"Global z={z}")

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
    train_curve = []
    val_curve = []
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
        depth_size=train_ds.crop_size,
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
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for cond2d, target3d, cond_indices in train_loader:
            # cond2d shape: [B, K, H, W]
            # target3d shape: [B, 1, D, H, W]
            cond2d = cond2d.to(device, non_blocking=True)
            target3d = target3d.to(device, non_blocking=True)

            b = target3d.shape[0]
            t = torch.randint(0, args.timesteps, (b,), device=device, dtype=torch.long)
            # t shape: [B]

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=torch.cuda.is_available()):
                loss = diffusion.p_losses(target3d, t, cond2d)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            update_ema(ema_model, model, decay=args.ema_decay)
            train_losses.append(loss.item())

        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        train_curve.append(mean_train)

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
                with autocast("cuda", enabled=torch.cuda.is_available()):
                    val_loss = diffusion.p_losses(target3d, t, cond2d)
                val_losses.append(val_loss.item())

        mean_val = float(np.mean(val_losses)) if val_losses else 0.0
        val_curve.append(mean_val)
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
            cond_ex, _, cond_indices_ex = val_ds[0]
            cond_ex = cond_ex.unsqueeze(0).to(device)

            crop = train_ds.crop_size

            sample = diffusion.sample(
                ema_model,
                cond_ex,
                shape=(1, 1, crop, crop, crop),
            )

            montage_path = os.path.join(
                args.save_dir, f"sample_epoch_{epoch:04d}.png"
            )
            save_montage(cond_ex[0], sample[0], cond_indices_ex, montage_path)

            render_path = os.path.join(
                args.save_dir, f"sample_3d_epoch_{epoch:04d}.png"
            )
            save_3d_render(sample[0], render_path)
            
    plt.figure()
    plt.plot(train_curve, label="train")
    plt.plot(val_curve, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "loss_curve.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
