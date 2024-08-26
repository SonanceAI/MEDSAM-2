import torch
from torch.utils.data import Dataset
import os
import cv2
from .sam_dataset import split_masks


class USThyroidDataset(Dataset):
    def __init__(self, root_dir: str,
                 split: str = 'All') -> None:
        super().__init__()
        self.root_dir = root_dir

        split = split.lower()
        if split not in ['all', 'train', 'test']:
            raise ValueError(f"split must be one of ['all', 'train', 'test'], got {split}")

        # tn3k folder
        tn3k_dir = os.path.join(root_dir, 'tn3k')
        tn3k_train_images = [f'tn3k/trainval-image/{fname}' for fname in os.listdir(f'{tn3k_dir}/trainval-image')
                             if fname.endswith('.jpg')]
        tn3k_test_images = [f'tn3k/test-image/{fname}' for fname in os.listdir(f'{tn3k_dir}/test-image')
                            if fname.endswith('.jpg')]
        tn3k_train_masks = [f'tn3k/trainval-mask/{fname}' for fname in os.listdir(f'{tn3k_dir}/trainval-mask')
                            if fname.endswith('.jpg')]
        tn3k_test_masks = [f'tn3k/test-mask/{fname}' for fname in os.listdir(f'{tn3k_dir}/test-mask')
                           if fname.endswith('.jpg')]

        # tg3k folder
        tg3k_dir = os.path.join(root_dir, 'tg3k')
        tg3k_images = [f'tg3k/thyroid-image/{fname}' for fname in os.listdir(f'{tg3k_dir}/thyroid-image')
                       if fname.endswith('.jpg')]
        tg3k_masks = [f'tg3k/thyroid-mask/{fname}' for fname in os.listdir(f'{tg3k_dir}/thyroid-mask')
                      if fname.endswith('.jpg')]
        
        if split == 'train':
            self.image_names = tn3k_train_images + tg3k_images
            self.mask_names = tn3k_train_masks + tg3k_masks
        elif split == 'test':
            self.image_names = tn3k_test_images
            self.mask_names = tn3k_test_masks
        else:
            self.image_names = tn3k_train_images + tn3k_test_images + tg3k_images
            self.mask_names = tn3k_train_masks + tn3k_test_masks + tg3k_masks

        self.image_names.sort()
        self.mask_names.sort()

        assert len(self.image_names) == len(self.mask_names), \
            f"Number of images and masks must be equal {len(self.image_names)} != {len(self.mask_names)}"

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

        # threshold the mask by 200 to remove weird letters burned into the masks
        gt_frame = (gt_frame >= 200).to(dtype=torch.uint8)

        # for each unique value in the mask, create a new mask
        gt_frame = split_masks(gt_frame)

        return {'image': img,
                'masks': gt_frame}
