import glob
import os
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


torch.manual_seed(42)
np.random.seed(42)


class BraTSDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        k_slices: int = 5,
        seed: int = 42,
        modalities: List[str] = None,
        cond_modality: str = "t1",
        crop_size: int = 128,
        patch_overlap: bool = False,
        overlap_stride: int = 64,
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if k_slices % 2 == 0:
            raise ValueError("k_slices must be odd")

        self.root_dir = root_dir
        self.split = split
        self.k_slices = k_slices
        self.seed = seed
        self.crop_size = crop_size
        self.modalities = modalities if modalities is not None else ["t1"]
        self.cond_modality = cond_modality
        self.patch_overlap = patch_overlap
        self.overlap_stride = overlap_stride

        if self.cond_modality not in self.modalities:
            raise ValueError("cond_modality must be included in modalities")

        self.subjects = self._collect_subjects()
        if len(self.subjects) == 0:
            raise RuntimeError(f"No valid BraTS subjects found under {root_dir}")

        rng = np.random.default_rng(seed)
        order = np.arange(len(self.subjects))
        rng.shuffle(order)
        n_train = int(0.8 * len(order))
        selected = order[:n_train] if split == "train" else order[n_train:]
        self.subjects = [self.subjects[i] for i in selected]

    def _collect_subjects(self) -> List[Dict[str, str]]:
        base = os.path.join(
            self.root_dir,
            "BraTS2020_TrainingData",
            "MICCAI_BraTS2020_TrainingData",
            "BraTS20_Training_*",
        )
        subject_dirs = sorted(glob.glob(base))
        subjects: List[Dict[str, str]] = []

        for sub_dir in subject_dirs:
            sid = os.path.basename(sub_dir)
            paths = {}
            ok = True
            for m in self.modalities:
                p_nii = os.path.join(sub_dir, f"{sid}_{m}.nii")
                p_niigz = os.path.join(sub_dir, f"{sid}_{m}.nii.gz")
                if os.path.exists(p_nii):
                    paths[m] = p_nii
                elif os.path.exists(p_niigz):
                    paths[m] = p_niigz
                else:
                    ok = False
                    break
            if ok:
                subjects.append(paths)
        return subjects

    def __len__(self) -> int:
        return len(self.subjects)

    @staticmethod
    def _load_canonical(path: str) -> np.ndarray:
        img = nib.load(path)
        img = nib.as_closest_canonical(img)
        vol = img.get_fdata().astype(np.float32)
        # [H, W, D] -> [D, H, W]
        vol = np.transpose(vol, (2, 0, 1))
        return vol

    @staticmethod
    def _normalize(vol: np.ndarray) -> np.ndarray:
        p_low = np.percentile(vol, 0.5)
        p_high = np.percentile(vol, 99.5)
        vol = np.clip(vol, p_low, p_high)
        mean = vol.mean(dtype=np.float32)
        std = vol.std(dtype=np.float32)
        return ((vol - mean) / (std + 1e-6)).astype(np.float32)

    def _crop_coords(self, shape: Tuple[int, int, int], idx: int) -> Tuple[int, int, int]:
        d, h, w = shape
        c = self.crop_size
        if d < c or h < c or w < c:
            raise ValueError(f"Volume too small for {c}^3 crop: {shape}")

        max_d, max_h, max_w = d - c, h - c, w - c
        if self.split != "train":
            return max_d // 2, max_h // 2, max_w // 2

        worker = get_worker_info()
        wid = 0 if worker is None else worker.id
        rng = np.random.default_rng(self.seed + idx + wid * 1000003)

        if self.patch_overlap:
            sd = min(self.overlap_stride, c)
            d_candidates = np.arange(0, max_d + 1, sd, dtype=np.int64)
            h_candidates = np.arange(0, max_h + 1, sd, dtype=np.int64)
            w_candidates = np.arange(0, max_w + 1, sd, dtype=np.int64)
            ds = int(d_candidates[rng.integers(0, len(d_candidates))]) if len(d_candidates) > 0 else 0
            hs = int(h_candidates[rng.integers(0, len(h_candidates))]) if len(h_candidates) > 0 else 0
            ws = int(w_candidates[rng.integers(0, len(w_candidates))]) if len(w_candidates) > 0 else 0
            return ds, hs, ws

        ds = int(rng.integers(0, max_d + 1)) if max_d > 0 else 0
        hs = int(rng.integers(0, max_h + 1)) if max_h > 0 else 0
        ws = int(rng.integers(0, max_w + 1)) if max_w > 0 else 0
        return ds, hs, ws

    def __getitem__(self, idx: int):
        sample = self.subjects[idx]

        vols = []
        for m in self.modalities:
            vol = self._load_canonical(sample[m])
            vol = self._normalize(vol)
            vols.append(vol)

        # list of [D, H, W] -> [C, D, H, W]
        vol4d = np.stack(vols, axis=0)
        c, d, h, w = vol4d.shape

        ds, hs, ws = self._crop_coords((d, h, w), idx)
        de, he, we = ds + self.crop_size, hs + self.crop_size, ws + self.crop_size
        vol4d = vol4d[:, ds:de, hs:he, ws:we]
        # vol4d: [C, 128, 128, 128]

        cond_idx = self.modalities.index(self.cond_modality)
        cond_src = vol4d[cond_idx]
        # cond_src: [D, H, W]
        z0 = cond_src.shape[0] // 2
        half = self.k_slices // 2
        z_idx = np.arange(z0 - half, z0 + half + 1)
        cond2d_np = np.stack([cond_src[z] for z in z_idx], axis=0)
        # cond2d_np: [K, H, W]

        target3d = torch.from_numpy(vol4d).float()
        # target3d: [C, D, H, W]
        cond2d = torch.from_numpy(cond2d_np).float()
        # cond2d: [K, H, W]
        return cond2d, target3d
