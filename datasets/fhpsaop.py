import torch
from torch.utils.data import Dataset
import os
from .sam_dataset import split_masks
import SimpleITK as sitk


class FHPSAOPDataset(Dataset):
    def __init__(self, root_dir: str,
                 split: str = 'All') -> None:
        super().__init__()
        self.root_dir = f'{root_dir}/Pubic Symphysis-Fetal Head Segmentation and Angle of Progression'

        split = split.lower()
        if split not in ['all', 'train', 'test']:
            raise ValueError(f"split must be one of ['all', 'train', 'test'], got {split}")

        self.image_names = [f'image_mha/{fname}' for fname in os.listdir(f'{self.root_dir}/image_mha')
                            if fname.endswith('.mha')]
        self.mask_names = [f'label_mha/{fname}' for fname in os.listdir(f'{self.root_dir}/label_mha')
                           if fname.endswith('.mha')]

        assert len(self.image_names) > 0
        assert len(self.image_names) == len(self.mask_names)

        self.image_names.sort()
        self.mask_names.sort()
        idx_split = len(self.image_names) * 4 // 5
        if split.lower() == 'train':
            self.image_names = self.image_names[:idx_split]
            self.mask_names = self.mask_names[:idx_split]
        elif split.lower() == 'test':
            self.image_names = self.image_names[idx_split:]
            self.mask_names = self.mask_names[idx_split:]

        

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index) -> dict:
        imgname = self.image_names[index]
        maskname = self.mask_names[index]

        # read data
        img = sitk.ReadImage(os.path.join(self.root_dir, imgname))
        img = sitk.GetArrayFromImage(img)
        masks = sitk.ReadImage(os.path.join(self.root_dir, maskname))
        masks = sitk.GetArrayFromImage(masks)

        # convert to tensors
        img = torch.tensor(img, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.uint8)

        # for each unique value in the mask, create a new mask
        masks = split_masks(masks)

        return {'image': img,
                'masks': masks}
