import argparse
import copy
import os
from dataclasses import dataclass

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset_brats import BraTSDataset
from diffusion3d import Diffusion3D
from unet3d_conditional import UNet3DConditional


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            else:
                v.copy_(msd[k])


@dataclass
class TrainConfig:
    root_dir: str
    output_dir: str = "checkpoints_brats"
    epochs: int = 100
    batch_size: int = 1
    num_workers: int = 2
    lr: float = 1e-4
    save_every: int = 5
    ema_decay: float = 0.999
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


def save_checkpoint(path: str, model: torch.nn.Module, ema: EMA, optimizer: torch.optim.Optimizer, scaler: GradScaler, epoch: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema.ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )


def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BraTSDataset(cfg.root_dir)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = UNet3DConditional(base_dim=32, depth_levels=3, max_depth=128).to(device)
    diffusion = Diffusion3D(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        device=device,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    ema = EMA(model, decay=cfg.ema_decay)

    os.makedirs(cfg.output_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        for cond2d, target3d in loader:
            # cond2d shape: [B, 1, H=128, W=128]
            # target3d shape: [B, 1, D=128, H=128, W=128]
            cond2d = cond2d.to(device, non_blocking=True)
            # cond2d shape: [B, 1, 128, 128]
            target3d = target3d.to(device, non_blocking=True)
            # target3d shape: [B, 1, 128, 128, 128]

            b = target3d.shape[0]
            t = diffusion.sample_timesteps(b, device)
            # t shape: [B]

            with autocast(enabled=torch.cuda.is_available()):
                x_t, noise = diffusion.forward_diffusion(target3d, t)
                # x_t shape: [B, 1, 128, 128, 128]
                # noise shape: [B, 1, 128, 128, 128]
                pred_noise = model(x_t, t, cond2d)
                # pred_noise shape: [B, 1, 128, 128, 128]
                loss = torch.nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running_loss += float(loss.detach().item())

        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch}/{cfg.epochs} | loss={avg_loss:.6f}")

        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            ckpt_path = os.path.join(cfg.output_dir, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(ckpt_path, model, ema, optimizer, scaler, epoch)
            print(f"Saved checkpoint: {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints_brats")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    args = parser.parse_args()
    return TrainConfig(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        save_every=args.save_every,
        ema_decay=args.ema_decay,
    )


if __name__ == "__main__":
    train(parse_args())
