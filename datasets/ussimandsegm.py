import torch
from torch.utils.data import Dataset
import os
import cv2
from .sam_dataset import split_masks
import numpy as np


class USsimandsegmDataset(Dataset):
    def __init__(self, root_dir: str,
                 split: str = 'All') -> None:
        super().__init__()
        root_dir_aus = os.path.join(root_dir, 'abdominal_US/abdominal_US/AUS')
        self.images_dir = os.path.join(root_dir_aus, 'images')
        self.masks_dir = os.path.join(root_dir_aus, 'annotations')

        split = split.lower()
        if split not in ['all', 'train', 'test']:
            raise ValueError(f"split must be one of ['all', 'train', 'test'], got {split}")

        train_image_names = [f'train/{fname}' for fname in os.listdir(f'{self.images_dir}/train')
                             if fname.endswith('.png')]
        test_image_names = [f'test/{fname}' for fname in os.listdir(f'{self.images_dir}/test')
                            if fname.endswith('.png')]

        if split == 'train':
            self.image_names = train_image_names
        elif split == 'test':
            self.image_names = test_image_names
        else:
            self.image_names = train_image_names + test_image_names

        self._remove_zero_masks()
        assert len(self.image_names) > 0

    def _remove_zero_masks(self):
        zero_masks = []
        for imgname in self.image_names:
            mask = cv2.imread(os.path.join(self.masks_dir, imgname),
                              cv2.IMREAD_GRAYSCALE)
            if not np.any(mask):
                zero_masks.append(os.path.basename(imgname))

        assert len(zero_masks) > 0
        self.image_names = [img_name for img_name in self.image_names
                            if os.path.basename(img_name) not in zero_masks]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index) -> dict:
        imgname = self.image_names[index]
        img = cv2.imread(os.path.join(self.images_dir, imgname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_frame = cv2.imread(os.path.join(self.masks_dir, imgname))

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        gt_frame = torch.tensor(gt_frame, dtype=torch.uint8)

        # for each unique value in the mask, create a new mask
        gt_frame = split_masks(gt_frame, black_color=30)

        return {'image': img,
                'masks': gt_frame}
