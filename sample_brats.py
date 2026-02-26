import argparse
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from dataset_brats import BraTSDataset
from diffusion3d import GaussianDiffusion3D
from unet3d_conditional import ConditionalUNet3D


torch.manual_seed(42)
np.random.seed(42)


def preprocess_t1(path: str, k_slices: int = 5):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    vol = img.get_fdata().astype(np.float32)
    # vol shape from nib: [H, W, D]
    vol = np.transpose(vol, (2, 0, 1))
    # vol shape: [D, H, W]

    d, h, w = vol.shape
    c = 128
    if d < c or h < c or w < c:
        raise ValueError(f"Volume too small for crop: {vol.shape}")

    d_start = (d - c) // 2
    d_end = d_start + c
    h_start = (h - c) // 2
    h_end = h_start + c
    w_start = (w - c) // 2
    w_end = w_start + c
    vol = vol[d_start:d_end, h_start:h_end, w_start:w_end]
    # vol shape: [128, 128, 128]

    p_low = np.percentile(vol, 0.5)
    p_high = np.percentile(vol, 99.5)
    vol = np.clip(vol, p_low, p_high)
    vol = (vol - vol.mean(dtype=np.float32)) / (vol.std(dtype=np.float32) + 1e-6)
    # vol shape: [128, 128, 128]

    z0 = vol.shape[0] // 2
    half = k_slices // 2
    cond = np.stack([vol[z] for z in range(z0 - half, z0 + half + 1)], axis=0)
    # cond shape: [K, H, W]
    return cond.astype(np.float32)


def save_png_montage(vol_dhw: np.ndarray, out_png: str):
    # vol_dhw shape: [D, H, W]
    z_list = [32, 64, 96]
    fig, axes = plt.subplots(1, len(z_list), figsize=(12, 4))
    for i, z in enumerate(z_list):
        axes[i].imshow(vol_dhw[z], cmap="gray")
        axes[i].set_title(f"z={z}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--subject_t1", type=str, default="")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./samples_brats")
    parser.add_argument("--use_ema", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]

    model = ConditionalUNet3D(
        in_channels=1,
        out_channels=1,
        k_slices=config["k_slices"],
        base_dim=config["base_dim"],
        levels=config["levels"],
        groupnorm_groups=8,
        depth_size=128,
    ).to(device)

    if args.use_ema and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion3D(
        model=model,
        timesteps=config["timesteps"],
        beta_start=1e-4,
        beta_end=0.02,
        objective=config.get("objective", "v"),
    ).to(device)

    if args.subject_t1:
        cond_np = preprocess_t1(args.subject_t1, k_slices=config["k_slices"])
        # cond_np shape: [K, H, W]
        cond2d = torch.from_numpy(cond_np).unsqueeze(0).to(device)
        # cond2d shape: [1, K, H, W]
    else:
        ds = BraTSDataset(args.root_dir, split=args.split, k_slices=config["k_slices"], seed=42)
        cond2d, _ = ds[args.index]
        # cond2d shape: [K, H, W]
        cond2d = cond2d.unsqueeze(0).to(device)
        # cond2d shape: [1, K, H, W]

    sample = diffusion.sample(model, cond2d, shape=(1, 1, 128, 128, 128))
    # sample shape: [1, 1, D, H, W]
    sample_dhw = sample[0, 0].detach().cpu().numpy().astype(np.float32)
    # sample_dhw shape: [D, H, W]

    nii_data = np.transpose(sample_dhw, (1, 2, 0))
    # nii_data shape: [H, W, D]
    nii = nib.Nifti1Image(nii_data, affine=np.eye(4, dtype=np.float32))
    nii_out = os.path.join(args.out_dir, "sample_volume.nii.gz")
    nib.save(nii, nii_out)

    png_out = os.path.join(args.out_dir, "sample_montage.png")
    save_png_montage(sample_dhw, png_out)

    print(f"Saved NIfTI: {nii_out}")
    print(f"Saved PNG: {png_out}")


if __name__ == "__main__":
    main()
