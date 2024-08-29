import torch
from torch.utils.data import Dataset
import os
import numpy as np


class FetalAbnominal(Dataset):
    def __init__(self, root_dir: str,
                 split: str = 'All') -> None:
        super().__init__()
        self.root_dir = os.path.join(root_dir,
                                     'Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images',
                                     'Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images',
                                     'ARRAY_FORMAT')

        if split.lower() not in ['all', 'train', 'test']:
            raise ValueError(f"split must be one of ['all', 'train', 'test'], got {split}")

        self.data_names = [fname for fname in os.listdir(self.root_dir)
                           if fname.endswith('.npy')]

        # sort the patients directories
        self.data_names.sort()
        idx_split = len(self.data_names) * 4 // 5
        if split.lower() == 'train':
            self.data_names = self.data_names[:idx_split]
        elif split.lower() == 'test':
            self.data_names = self.data_names[idx_split:]

        assert len(self.data_names) > 0

    def __len__(self) -> int:
        return len(self.data_names)

    def __getitem__(self, index) -> dict:
        dataname = self.data_names[index]
        data = np.load(f'{self.root_dir}/{dataname}', allow_pickle=True).item()
        img = data['image']  # uint8 - 0 to 255
        if img.ndim == 2:
            # (H, W) -> (C, H, W)
            img = np.stack([img] * 3, axis=0)
            img = torch.tensor(img, dtype=torch.float32)
        else:
            # (H, W, C) -> (C, H, W)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        gt_frame = np.array(list(data['structures'].values()), dtype=np.uint8)
        gt_frame = torch.from_numpy(gt_frame)

        assert gt_frame.max() == 1

        return {'image': img,
                'masks': gt_frame,
                'image_name': dataname}
