import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np
from .sam_dataset import split_masks


class CAMUS(Dataset):
    def __init__(self, root_dir: str,
                 split: str = 'All') -> None:
        super().__init__()
        self.root_dir = root_dir

        if split.lower() not in ['all', 'train', 'test']:
            raise ValueError(f"split must be one of ['all', 'train', 'test'], got {split}")

        # list all directories in f'{root_dir}/CAMUS_public/database_nifti'
        self.patients_dirs = os.listdir(f'{root_dir}/CAMUS_public/database_nifti')
        self.patients_dirs = [p for p in self.patients_dirs if p.startswith('patient')]

        # sort the patients directories
        self.patients_dirs.sort()
        idx_split = len(self.patients_dirs) * 4 // 5
        if split.lower() == 'train':
            self.patients_dirs = self.patients_dirs[:idx_split]
        elif split.lower() == 'test':
            self.patients_dirs = self.patients_dirs[idx_split:]

        self.nii_files = []
        for patient in self.patients_dirs:
            pdir = f'{root_dir}/CAMUS_public/database_nifti/{patient}'
            self.nii_files += [f'{patient}/{f}' for f in os.listdir(pdir)
                               if f.endswith('.nii.gz') and 'half_sequence' in f and not f.endswith('_gt.nii.gz')]

        # get number of frames
        self.n_frames = []
        for f in self.nii_files:
            self.n_frames.append(nib.load(f'{root_dir}/CAMUS_public/database_nifti/{f}').shape[-1])

    def __len__(self) -> int:
        return sum(self.n_frames)

    def _load_frame_by_idx(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        # get the file path
        frame_idx = 0
        for i, n in enumerate(self.n_frames):
            if idx < n:
                frame_idx = idx
                break
            idx -= n

        file_path = f'{self.root_dir}/CAMUS_public/database_nifti/{self.nii_files[i]}'
        file_path_gt = file_path.replace('half_sequence', 'half_sequence_gt')
        # load the file
        img = nib.load(file_path).get_fdata()
        assert img.ndim == 3
        img = img[:, :, frame_idx]
        gt_frame = nib.load(file_path_gt).get_fdata()
        gt_frame = gt_frame[..., frame_idx]

        return img, gt_frame

    def __getitem__(self, index) -> dict:
        img, gt_frame = self._load_frame_by_idx(index)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        gt_frame = torch.tensor(gt_frame, dtype=torch.uint8)

        # (1, H, W) to (3, H, W)
        img = torch.cat([img, img, img], dim=0)

        # for each unique value in the mask, create a new mask
        gt_frame = split_masks(gt_frame)
        

        return {'image': img,
                'masks': gt_frame}
