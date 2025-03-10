from torch.utils.data import DataLoader, Dataset
import os
from lightning import LightningDataModule
from torchvision import transforms as T
import torch
import cv2
import numpy as np
from typing import Any


class LabPicsDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 folder_name: str = 'Train',
                 transform=None):
        self.data = self._scan_data(data_dir, folder_name)
        self.mask_transform = T.Compose([
            T.ToTensor(),
            # T.Resize((1024, 1024), interpolation=T.InterpolationMode.NEAREST, antialias=False)
        ])

        if transform is None:
            self.transform = T.ToTensor()
        else:
            self.transform = transform

    def _scan_data(self, data_dir: str, folder_name: str) -> list[dict]:
        data = []
        for ff, name in enumerate(os.listdir(f"{data_dir}/Simple/{folder_name}/Image/")):
            data.append({"image": f"{data_dir}/Simple/{folder_name}/Image/"+name,
                        "annotation": f"{data_dir}/Simple/{folder_name}/Instance/"+name[:-4]+".png"})
        return data

    def _read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # read random image and its annotaion from  the dataset (LabPics)
    def _read_single_image(self, ent: dict) -> dict[str, Any]:
        #  select image
        img = self._read_image(ent["image"])
        ann_map = self._read_image(ent["annotation"])
        # resize image

        img = self.transform(img)
        ann_map = self.mask_transform(ann_map)
        ann_map = ann_map.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        # r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
        # img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
        # ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r),
        #                                int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

        # merge vessels and materials annotations
        mat_map = ann_map[:, :, 0]
        ves_map = ann_map[:, :, 2]
        mat_map[mat_map == 0] = ves_map[mat_map == 0]*(mat_map.max()+1)

        # Get binary masks and points
        inds = np.unique(mat_map)[1:]
        points = []
        masks = []
        for ind in inds:
            mask = (mat_map == ind).to(dtype=torch.uint8)
            masks.append(mask)
            coords = torch.argwhere(mask > 0)
            yx = coords[np.random.randint(len(coords))]
            points.append([[yx[1], yx[0]]])
        # img.shape: (H,W,C)
        if len(masks) == 0:
            masks = torch.zeros(size=(0,))
            points = torch.zeros(size=(0,))
        else:
            masks = torch.stack(masks)
            points = torch.tensor(points)

        # masks.shape: (N, H, W)
        return {
            'image': img,
            'masks': masks,
            'points_coords': points,
            'points_labels': torch.ones((len(points), 1)),
            'image_name': ent["image"]
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict[str, Any]:
        return self._read_single_image(self.data[idx])


def collate_dict_SAM(batch: list[dict[str, Any]]) -> dict:
    out_dict = {'image': torch.stack([item['image'] for item in batch])}
    for key in ('masks', 'points_coords', 'points_labels', 'boxes'):
        if key in batch[0]:
            out_dict[key] = [item[key] for item in batch]

    return out_dict


class LabPicsDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage: str = None):
        self.train_dataset = LabPicsDataset(self.data_dir, 'Train', transform=self.transform)
        self.val_dataset = LabPicsDataset(self.data_dir, 'Test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=6,
                          collate_fn=collate_dict_SAM
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=6,
                          collate_fn=collate_dict_SAM)
