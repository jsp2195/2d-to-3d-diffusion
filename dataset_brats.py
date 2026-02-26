import glob
import os
import random
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


torch.manual_seed(42)
np.random.seed(42)


class BraTSDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", k_slices: int = 5, seed: int = 42):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if k_slices % 2 == 0:
            raise ValueError("k_slices must be odd")

        self.root_dir = root_dir
        self.split = split
        self.k_slices = k_slices
        self.seed = seed
        self.crop_size = 128

        self.paths = self._collect_t1_paths(root_dir)
        if len(self.paths) == 0:
            raise RuntimeError(f"No T1 files found under {root_dir}")

        rng = random.Random(seed)
        shuffled = self.paths.copy()
        rng.shuffle(shuffled)
        n_train = int(0.8 * len(shuffled))
        self.split_paths = shuffled[:n_train] if split == "train" else shuffled[n_train:]

    @staticmethod
    def _collect_t1_paths(root_dir: str) -> List[str]:
        pattern = os.path.join(
            root_dir,
            "BraTS2020_TrainingData",
            "MICCAI_BraTS2020_TrainingData",
            "BraTS20_Training_*",
            "*_t1.nii*",
        )
        candidates = sorted(glob.glob(pattern))
        t1_paths = []
        for p in candidates:
            lower = p.lower()
            if lower.endswith("_t1.nii") or lower.endswith("_t1.nii.gz"):
                t1_paths.append(p)
        return t1_paths

    def __len__(self) -> int:
        return len(self.split_paths)

    def _center_crop_3d(self, vol: np.ndarray) -> np.ndarray:
        # vol shape: [D, H, W]
        d, h, w = vol.shape
        c = self.crop_size
        if d < c or h < c or w < c:
            raise ValueError(f"Volume too small for 128^3 crop: got {vol.shape}")

        d_start = (d - c) // 2
        d_end = d_start + c
        h_start = (h - c) // 2
        h_end = h_start + c
        w_start = (w - c) // 2
        w_end = w_start + c

        # cropped shape: [128, 128, 128]
        return vol[d_start:d_end, h_start:h_end, w_start:w_end]

    @staticmethod
    def _normalize(vol: np.ndarray) -> np.ndarray:
        # vol shape: [D, H, W]
        p_low = np.percentile(vol, 0.5)
        p_high = np.percentile(vol, 99.5)
        vol = np.clip(vol, p_low, p_high)
        mean = vol.mean(dtype=np.float32)
        std = vol.std(dtype=np.float32)
        vol = (vol - mean) / (std + 1e-6)
        return vol.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        path = self.split_paths[idx]

        img = nib.load(path)
        img = nib.as_closest_canonical(img)
        vol = img.get_fdata().astype(np.float32)
        # vol shape from nib: [H, W, D]
        vol = np.transpose(vol, (2, 0, 1))
        # vol shape after transpose: [D, H, W]

        vol = self._center_crop_3d(vol)
        # vol shape: [128, 128, 128]
        vol = self._normalize(vol)
        # vol shape: [128, 128, 128]

        d, h, w = vol.shape
        z0 = d // 2
        half = self.k_slices // 2
        indices = list(range(z0 - half, z0 + half + 1))

        cond2d_np = np.stack([vol[z] for z in indices], axis=0)
        # cond2d_np shape: [K, H, W]
        target3d_np = vol[np.newaxis, ...]
        # target3d_np shape: [1, D, H, W]

        cond2d = torch.from_numpy(cond2d_np).float()
        # cond2d shape: [K, H, W]
        target3d = torch.from_numpy(target3d_np).float()
        # target3d shape: [1, D, H, W]
        return cond2d, target3d
