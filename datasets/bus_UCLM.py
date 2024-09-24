import torch
from torch.utils.data import Dataset
import os
import cv2
from .sam_dataset import split_masks
import numpy as np


class BUS_UCLM(Dataset):
    def __init__(self, root_dir: str,
                 split: str = 'All') -> None:
        super().__init__()
        self.root_dir = os.path.join(root_dir,
                                     'BUS-UCLM Breast ultrasound lesion segmentation dataset',
                                     'BUS-UCLM')

        if split.lower() not in ['all', 'train', 'test']:
            raise ValueError(f"split must be one of ['all', 'train', 'test'], got {split}")

        self.image_names = [f'images/{fname}' for fname in os.listdir(f'{self.root_dir}/images')
                            if fname.endswith('.png')]

        self.mask_names = [f'masks/{fname}' for fname in os.listdir(f'{self.root_dir}/masks')
                           if fname.endswith('.png')]

        # sort the patients directories
        self.image_names.sort()
        self.mask_names.sort()
        idx_split = len(self.image_names) * 4 // 5
        if split.lower() == 'train':
            self.image_names = self.image_names[:idx_split]
            self.mask_names = self.mask_names[:idx_split]
        elif split.lower() == 'test':
            self.image_names = self.image_names[idx_split:]
            self.mask_names = self.mask_names[idx_split:]

        self._remove_zero_masks()

        assert len(self.image_names) > 0
        assert len(self.image_names) == len(self.mask_names)

    def _remove_zero_masks(self):
        zero_masks = []
        for mask_name in self.mask_names:
            mask = cv2.imread(f'{self.root_dir}/{mask_name}', cv2.IMREAD_GRAYSCALE)
            if not np.any(mask):
                zero_masks.append(os.path.basename(mask_name))

        assert len(zero_masks) > 0

        self.image_names = [img_name for img_name in self.image_names
                            if os.path.basename(img_name) not in zero_masks]
        self.mask_names = [mask_name for mask_name in self.mask_names
                           if os.path.basename(mask_name) not in zero_masks]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index) -> dict:
        imgname = self.image_names[index]
        img = cv2.imread(f'{self.root_dir}/{imgname}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        maskname = self.mask_names[index]
        gt_frame = cv2.imread(f'{self.root_dir}/{maskname}', cv2.IMREAD_GRAYSCALE)
        gt_frame = (gt_frame > 127).astype(np.uint8)
        # TODO: pre-compute connectedComponents and save it somewhere
        _, gt_frame = cv2.connectedComponents(gt_frame)

        # (H, W, C) -> (C, H, W)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        gt_frame = torch.tensor(gt_frame, dtype=torch.uint8)

        # for each unique value in the mask, create a new mask
        gt_frame = split_masks(gt_frame)

        return {'image': img,
                'masks': gt_frame,
                'image_name': imgname}
