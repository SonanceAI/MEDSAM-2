import torch
from torch.utils.data import Dataset
import os
import cv2
from .sam_dataset import split_masks


class BUSI_WHU(Dataset):
    def __init__(self, root_dir: str,
                 split: str = 'All') -> None:
        super().__init__()
        self.root_dir = os.path.join(root_dir,
                                     'BUSI_WHU Breast Cancer Ultrasound Image  Dataset',
                                     'BUSI_WHU')

        if split.lower() not in ['all', 'train', 'test', 'valid']:
            raise ValueError(f"split must be one of ['all', 'train', 'test', 'valid'], got {split}")

        image_names = {}

        for spl_folder in ['train', 'test', 'valid']:
            imnames = [f'{spl_folder}/img/{fname}' for fname in os.listdir(f'{self.root_dir}/{spl_folder}/img')
                       if fname.endswith('.png')]
            image_names[spl_folder] = imnames

        if split.lower() == 'all':
            self.image_names = image_names['train'] + image_names['test'] + image_names['valid']
        else:
            self.image_names = image_names[split.lower()]

        self.mask_names = [imgname.replace('/img/', '/gt/').replace('.bmp', '_anno.bmp')
                           for imgname in self.image_names]

        assert len(self.image_names) == len(self.mask_names)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index) -> dict:
        imgname = self.image_names[index]
        img = cv2.imread(f'{self.root_dir}/{imgname}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        maskname = self.mask_names[index]
        gt_frame = cv2.imread(f'{self.root_dir}/{maskname}', cv2.IMREAD_GRAYSCALE)

        # (H, W, C) -> (C, H, W)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        gt_frame = torch.tensor(gt_frame, dtype=torch.uint8)
        gt_frame = (gt_frame > 127).to(dtype=torch.uint8)

        # for each unique value in the mask, create a new mask
        gt_frame = split_masks(gt_frame)

        return {'image': img,
                'masks': gt_frame,
                'image_name': imgname}
