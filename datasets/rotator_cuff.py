import torch
from torch.utils.data import Dataset
import os
import cv2
from .sam_dataset import split_masks
from .dataset_utils import listdir


class RotatorCuffDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'All') -> None:
        super().__init__()

        split = split.lower()
        if split not in ['all', 'train', 'test']:
            raise ValueError(f"split must be one of ['all', 'train', 'test'], got {split}")

        self.root_dir = root_dir

        self.study_dirs = [fname for fname in os.listdir(f'{self.root_dir}')
                           if os.path.isdir(f'{self.root_dir}/{fname}')]
        self.study_dirs.sort()
        split_idx = int(len(self.study_dirs) * 0.8)
        if split == 'train':
            self.study_dirs = self.study_dirs[:split_idx]
        elif split == 'test':
            self.study_dirs = self.study_dirs[split_idx:]

        self.image_names_dict = {study_name: listdir(f'{self.root_dir}/{study_name}', endswith='_img.png')
                            for study_name in self.study_dirs}

        self.mask_names_dict = {study_name: [imgname.replace('_img.png', '_mask.png') for imgname in imgs]
                           for study_name, imgs in self.image_names_dict.items()}

        assert len(self.image_names_dict) == len(self.mask_names_dict)

        self.image_names = [f'{study_name}/{imgname}' for study_name in self.study_dirs for imgname in self.image_names_dict[study_name]]
        self.mask_names = [imgname.replace('_img.png','_mask.png') for imgname in self.image_names]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index) -> dict:
        imgname = self.image_names[index]
        maskname = self.mask_names[index]

        # read data
        img = cv2.imread(os.path.join(self.root_dir, imgname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_frame = cv2.imread(os.path.join(self.root_dir, maskname), cv2.IMREAD_GRAYSCALE)

        # convert to tensors
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        gt_frame = torch.tensor(gt_frame, dtype=torch.uint8)

        # for each unique value in the mask, create a new mask
        gt_frame = split_masks(gt_frame)

        return {'image': img,
                'masks': gt_frame,
                'image_name': imgname}
