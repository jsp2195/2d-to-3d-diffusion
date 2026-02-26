import argparse
import os

import nibabel as nib
import numpy as np
import torch

from dataset_brats import BraTSDataset
from diffusion3d import Diffusion3D
from unet3d_conditional import UNet3DConditional


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, use_ema: bool = True):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if use_ema and "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)


def sample_one(
    checkpoint_path: str,
    output_path: str,
    root_dir: str = None,
    index: int = 0,
    batch_size: int = 1,
    d: int = 128,
    h: int = 128,
    w: int = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3DConditional(base_dim=32, depth_levels=3, max_depth=128).to(device)
    load_checkpoint(model, checkpoint_path, use_ema=True)
    model.eval()

    diffusion = Diffusion3D(timesteps=1000, beta_start=1e-4, beta_end=2e-2, device=device).to(device)

    if root_dir is not None:
        dataset = BraTSDataset(root_dir)
        cond2d, _ = dataset[index]
        # cond2d shape: [1, H=128, W=128]
        cond2d = cond2d.unsqueeze(0).to(device)
        # cond2d shape: [B=1, 1, H=128, W=128]
    else:
        cond2d = torch.randn(batch_size, 1, h, w, device=device)
        # cond2d shape: [B, 1, H=128, W=128]

    with torch.no_grad():
        x = torch.randn(batch_size, 1, d, h, w, device=device)
        # x shape: [B, 1, D=128, H=128, W=128]
        for i in reversed(range(diffusion.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # t shape: [B]
            pred_noise = model(x, t, cond2d)
            # pred_noise shape: [B, 1, D=128, H=128, W=128]

            beta_t = diffusion.betas[t].view(-1, 1, 1, 1, 1)
            # beta_t shape: [B, 1, 1, 1, 1]
            alpha_t = diffusion.alphas[t].view(-1, 1, 1, 1, 1)
            # alpha_t shape: [B, 1, 1, 1, 1]
            sqrt_one_minus_ab_t = diffusion.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
            # sqrt_one_minus_ab_t shape: [B, 1, 1, 1, 1]
            sqrt_recip_alpha_t = diffusion.sqrt_recip_alpha[t].view(-1, 1, 1, 1, 1)
            # sqrt_recip_alpha_t shape: [B, 1, 1, 1, 1]

            model_mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_ab_t) * pred_noise)
            # model_mean shape: [B, 1, D=128, H=128, W=128]

            if i > 0:
                noise = torch.randn_like(x)
                # noise shape: [B, 1, D=128, H=128, W=128]
                posterior_var_t = diffusion.posterior_variance[t].view(-1, 1, 1, 1, 1)
                # posterior_var_t shape: [B, 1, 1, 1, 1]
                x = model_mean + torch.sqrt(posterior_var_t) * noise
                # x shape: [B, 1, D=128, H=128, W=128]
            else:
                x = model_mean
                # x shape: [B, 1, D=128, H=128, W=128]

    vol = x[0, 0].detach().cpu().numpy().astype(np.float32)
    # vol shape: [D=128, H=128, W=128]
    nib.save(nib.Nifti1Image(vol, affine=np.eye(4, dtype=np.float32)), output_path)
    print(f"Saved sample to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="sample.nii.gz")
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sample_one(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        root_dir=args.root_dir,
        index=args.index,
        batch_size=args.batch_size,
    )
