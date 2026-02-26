import glob
import os
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        pattern = os.path.join(
            root_dir,
            "MICCAI_BraTS2020_TrainingData",
            "BraTS20_Training_*",
            "*_t1.nii*",
        )
        self.t1_files: List[str] = sorted(glob.glob(pattern))
        if len(self.t1_files) == 0:
            raise FileNotFoundError(f"No T1 files found with pattern: {pattern}")

    def __len__(self) -> int:
        return len(self.t1_files)

    @staticmethod
    def _center_crop_3d(volume: np.ndarray, out_size: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
        # volume shape: [D, H, W]
        d, h, w = volume.shape
        out_d, out_h, out_w = out_size
        if d < out_d or h < out_h or w < out_w:
            raise ValueError(f"Input volume shape {volume.shape} is smaller than crop size {out_size}.")
        d0 = (d - out_d) // 2
        h0 = (h - out_h) // 2
        w0 = (w - out_w) // 2
        # cropped shape: [128, 128, 128]
        cropped = volume[d0:d0 + out_d, h0:h0 + out_h, w0:w0 + out_w]
        return cropped

    @staticmethod
    def _zscore_normalize(volume: np.ndarray) -> np.ndarray:
        # volume shape: [D, H, W]
        mean = float(volume.mean())
        std = float(volume.std())
        if std < 1e-6:
            std = 1.0
        # normalized shape: [D, H, W]
        normalized = (volume - mean) / std
        return normalized

    def __getitem__(self, index: int):
        t1_path = self.t1_files[index]
        nii = nib.load(t1_path)
        vol = nii.get_fdata(dtype=np.float32)
        # raw vol shape: [H, W, D]
        vol = np.transpose(vol, (2, 0, 1))
        # reordered vol shape: [D, H, W]
        vol = self._center_crop_3d(vol, (128, 128, 128))
        # cropped vol shape: [D=128, H=128, W=128]
        vol = self._zscore_normalize(vol).astype(np.float32)
        # normalized vol shape: [D=128, H=128, W=128]

        d = vol.shape[0]
        center_idx = d // 2
        cond2d = vol[center_idx]
        # cond2d shape: [H=128, W=128]
        target3d = vol
        # target3d shape: [D=128, H=128, W=128]

        cond2d_t = torch.from_numpy(cond2d).unsqueeze(0).float()
        # cond2d_t shape: [C=1, H=128, W=128]
        target3d_t = torch.from_numpy(target3d).unsqueeze(0).float()
        # target3d_t shape: [C=1, D=128, H=128, W=128]

        return cond2d_t, target3d_t


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    args = parser.parse_args()

    ds = BraTSDataset(args.root_dir)
    c, t = ds[0]
    print(c.shape, t.shape)
