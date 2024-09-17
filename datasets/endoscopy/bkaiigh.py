import torch
from torch.utils.data import Dataset
import os
from ..sam_dataset import split_masks
from ..dataset_utils import listdir
import cv2


class BKAI_IGH_Dataset(Dataset):
    def __init__(self, root_dir: str,
                 split: str = 'All') -> None:
        super().__init__()
        self.root_dir = root_dir
        if split.lower() not in ['all', 'train', 'test']:
            raise ValueError(f"split must be one of ['all', 'train', 'test'], got {split}")

        self.images_names = listdir(os.path.join(self.root_dir, 'train', 'train'), endswith='.jpeg')

        # sort the patients directories
        self.images_names.sort()
        idx_split = len(self.images_names) * 4 // 5
        if split.lower() == 'train':
            self.images_names = self.images_names[:idx_split]
        elif split.lower() == 'test':
            self.images_names = self.images_names[idx_split:]

        assert len(self.images_names) > 0

    def __len__(self) -> int:
        return len(self.images_names)

    def __getitem__(self, index) -> dict:
        imgname = self.images_names[index]
        img = cv2.imread(os.path.join(self.root_dir, 'train', 'train', imgname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.root_dir, 'train_gt', 'train_gt', imgname)
        gt_frame = cv2.imread(mask_path)

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        gt_frame = torch.tensor(gt_frame, dtype=torch.uint8)
        gt_frame = (gt_frame > 170).to(torch.uint8)

        # for each unique value in the mask, create a new mask
        gt_frame = split_masks(gt_frame)

        return {'image': img,
                'masks': gt_frame,
                'image_name': imgname}
