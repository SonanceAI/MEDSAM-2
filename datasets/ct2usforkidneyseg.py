import torch
from torch.utils.data import Dataset
import os
import cv2
from .sam_dataset import split_masks


class USForKidney(Dataset):
    def __init__(self, root_dir: str,
                 split: str = 'All') -> None:
        super().__init__()
        self.root_dir = root_dir

        if split.lower() not in ['all', 'train', 'test']:
            raise ValueError(f"split must be one of ['all', 'train', 'test'], got {split}")

        self.image_names = [fname for fname in os.listdir(f'{root_dir}/slice/slice/') if fname.endswith('.png')]

        # sort the patients directories
        self.image_names.sort()
        idx_split = len(self.image_names) * 4 // 5
        if split.lower() == 'train':
            self.image_names = self.image_names[:idx_split]
        elif split.lower() == 'test':
            self.image_names = self.image_names[idx_split:]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index) -> dict:
        imgname = self.image_names[index]
        img = cv2.imread(f'{self.root_dir}/slice/slice/{imgname}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_frame = cv2.imread(f'{self.root_dir}/mask/mask/{imgname}', cv2.IMREAD_GRAYSCALE)

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        gt_frame = torch.tensor(gt_frame, dtype=torch.uint8)

        # for each unique value in the mask, create a new mask
        gt_frame = split_masks(gt_frame)

        return {'image': img,
                'masks': gt_frame}
