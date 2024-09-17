import torch
from torch.utils.data import Dataset
import os
from ..sam_dataset import split_masks
from ..dataset_utils import listdir
import cv2


class PolypGen2021Dataset(Dataset):
    def __init__(self, root_dir: str,
                 split: str = 'All') -> None:
        super().__init__()
        self.root_dir = os.path.join(root_dir, 'PolypGen2021_MultiCenterData_v3')

        if split.lower() not in ['all', 'train', 'test']:
            raise ValueError(f"split must be one of ['all', 'train', 'test'], got {split}")

        self.images_path = []
        for i in range(1, 7):
            single_frame_dir = f'data_C{i}'
            images_dir = os.path.join(self.root_dir, single_frame_dir, f'images_C{i}')
            images_names_i = listdir(images_dir, endswith='.jpg', return_with_dir=True)
            assert len(images_names_i) > 0
            self.images_path.extend(images_names_i)

        # sequence dir
        seqs_dir = os.path.join(self.root_dir, 'sequenceData', 'positive')
        for seq_dir_i in listdir(seqs_dir, startswith='seq'):
            idx = int(seq_dir_i[3:])
            images_names_i = listdir(os.path.join(seqs_dir, seq_dir_i, f'images_seq{idx}'),
                                     endswith='.jpg', return_with_dir=True)
            assert len(images_names_i) > 0
            self.images_path.extend(images_names_i)

        # sort the patients directories
        self.images_path.sort()
        idx_split = len(self.images_path) * 4 // 5
        if split.lower() == 'train':
            self.images_path = self.images_path[:idx_split]
        elif split.lower() == 'test':
            self.images_path = self.images_path[idx_split:]

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, index) -> dict:
        img_path = self.images_path[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = img_path.replace('/images_seq', '/masks_seq').replace('/images_C', '/masks_C')
        mask_path = mask_path.replace('.jpg', '_mask.jpg')

        gt_frame = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        gt_frame = torch.tensor(gt_frame, dtype=torch.uint8)
        gt_frame = (gt_frame > 127).to(dtype=torch.uint8)

        # for each unique value in the mask, create a new mask
        gt_frame = split_masks(gt_frame)

        return {'image': img,
                'masks': gt_frame,
                'image_name': os.path.basename(img_path)}
