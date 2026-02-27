import argparse
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch.cuda.amp import autocast

from dataset_brats import BraTSDataset
from diffusion3d import GaussianDiffusion3D
from unet3d_conditional import ConditionalUNet3D


torch.manual_seed(42)
np.random.seed(42)


def preprocess_subject(path_map, modalities, cond_modality, k_slices=5, crop_size=128):
    vols = []
    for m in modalities:
        img = nib.load(path_map[m])
        img = nib.as_closest_canonical(img)
        vol = img.get_fdata().astype(np.float32)
        vol = np.transpose(vol, (2, 0, 1))
        p_low = np.percentile(vol, 0.5)
        p_high = np.percentile(vol, 99.5)
        vol = np.clip(vol, p_low, p_high)
        vol = (vol - vol.mean(dtype=np.float32)) / (vol.std(dtype=np.float32) + 1e-6)
        vols.append(vol)

    vol4d = np.stack(vols, axis=0)
    c, d, h, w = vol4d.shape

    ds = (d - crop_size) // 2
    hs = (h - crop_size) // 2
    ws = (w - crop_size) // 2
    vol4d = vol4d[:, ds:ds + crop_size, hs:hs + crop_size, ws:ws + crop_size]

    cond_src = vol4d[modalities.index(cond_modality)]
    z0 = cond_src.shape[0] // 2
    half = k_slices // 2
    cond = np.stack([cond_src[z] for z in range(z0 - half, z0 + half + 1)], axis=0)
    return cond.astype(np.float32), vol4d.astype(np.float32)


def save_montage(vol_dhw: np.ndarray, out_png: str):
    z_list = [32, 64, 96]
    fig, axes = plt.subplots(1, len(z_list), figsize=(12, 4))
    for i, z in enumerate(z_list):
        axes[i].imshow(vol_dhw[z], cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"z={z}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="./samples_brats")
    p.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--sampling_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--subject_dir", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    modalities = ckpt.get("modalities", cfg.get("modalities", ["t1", "t2", "flair", "t1ce"]))

    in_ch = len(modalities)
    model = ConditionalUNet3D(
        in_channels=in_ch,
        out_channels=in_ch * 2,
        k_slices=cfg["k_slices"],
        base_dim=cfg["base_dim"],
        levels=cfg["levels"],
        use_checkpoint=False,
    ).to(device)

    if "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion3D(
        model=model,
        channels=in_ch,
        timesteps=cfg["timesteps"],
        objective=cfg.get("objective", "v"),
        cfg_drop_prob=cfg.get("cfg_drop_prob", 0.1),
        vlb_weight=cfg.get("vlb_weight", 1e-3),
    ).to(device)

    if args.subject_dir:
        sid = os.path.basename(args.subject_dir)
        path_map = {}
        for m in modalities:
            p1 = os.path.join(args.subject_dir, f"{sid}_{m}.nii")
            p2 = os.path.join(args.subject_dir, f"{sid}_{m}.nii.gz")
            path_map[m] = p1 if os.path.exists(p1) else p2
        cond_np, _ = preprocess_subject(path_map, modalities, cfg.get("cond_modality", "t1"), k_slices=cfg["k_slices"])
        cond2d = torch.from_numpy(cond_np).unsqueeze(0).to(device)
    else:
        ds = BraTSDataset(
            root_dir=args.root_dir,
            split=args.split,
            k_slices=cfg["k_slices"],
            seed=42,
            modalities=modalities,
            cond_modality=cfg.get("cond_modality", "t1"),
        )
        cond2d, _ = ds[args.index]
        cond2d = cond2d.unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast(enabled=torch.cuda.is_available()):
            if args.sampler == "ddim":
                sample = diffusion.sample_ddim(
                    cond2d,
                    shape=(1, in_ch, 128, 128, 128),
                    sampling_steps=args.sampling_steps,
                    guidance_scale=args.guidance_scale,
                )
            else:
                sample = diffusion.sample(
                    cond2d,
                    shape=(1, in_ch, 128, 128, 128),
                    guidance_scale=args.guidance_scale,
                )

    vol = sample[0, 0].detach().cpu().numpy().astype(np.float32)
    nii_data = np.transpose(vol, (1, 2, 0))
    nii = nib.Nifti1Image(nii_data, affine=np.eye(4, dtype=np.float32))
    nii_path = os.path.join(args.out_dir, "sample_volume.nii.gz")
    nib.save(nii, nii_path)

    png_path = os.path.join(args.out_dir, "sample_montage.png")
    save_montage(vol, png_path)

    print(f"Saved NIfTI: {nii_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
