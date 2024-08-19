import torch
from torch.utils.data import Dataset
import os
import cv2
from .sam_dataset import split_masks


class USnervesegDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        super().__init__()
        self.root_dir = root_dir

        self.image_names = [f'train/{fname}' for fname in os.listdir(f'{self.root_dir}/train')
                            if fname.endswith('.tif') and not fname.endswith('_mask.tif') and os.path.isfile(f'{self.root_dir}/train/{fname[:-4]}_mask.tif')]

        self.mask_names = [f'{fname[:-4]}_mask.tif' for fname in self.image_names]

        # remove masks that are empty
        self.image_names = [img for img, mask in zip(self.image_names, self.mask_names)
                            if cv2.imread(os.path.join(self.root_dir, mask), cv2.IMREAD_GRAYSCALE).max() > 0]
        self.mask_names = [f'{fname[:-4]}_mask.tif' for fname in self.image_names]

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
                'masks': gt_frame}
