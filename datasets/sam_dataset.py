import torch
from torch.utils.data import Dataset
from typing import Any
import numpy as np
from torchvision import transforms as T
from sam2_modified import SAM2TransformsTensor


def generate_random_points(masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if len(masks) == 0:
        masks = torch.zeros(size=(0,))
        points = torch.zeros(size=(0,))
    else:
        points = []
        for mask in masks:
            coords = torch.argwhere(mask > 0)
            if len(coords) == 0:
                print(mask.sum())
                continue
            yx = coords[np.random.randint(len(coords))]
            points.append([[yx[1], yx[0]]])
        points = torch.tensor(points)

    return points, torch.ones((len(points), 1))


def collate_dict_SAM(batch: list[dict[str, Any]]) -> dict:
    out_dict = {'image': torch.stack([item['image'] for item in batch])}
    for key in ('masks', 'points_coords', 'points_labels', 'boxes'):
        if key in batch[0]:
            out_dict[key] = [item[key] for item in batch]

    return out_dict


class SAMDataset(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 image_transform: SAM2TransformsTensor) -> None:
        self.dataset = dataset
        self.mask_transform = T.Resize((1024, 1024),
                                       interpolation=T.InterpolationMode.NEAREST,
                                       antialias=False)
        assert isinstance(image_transform, SAM2TransformsTensor)
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> dict[str, Any]:
        data: dict = self.dataset[index]
        masks = data['masks']
        if masks.shape[0] != 0:
            masks = self.mask_transform(masks)
            masks = masks[masks.sum((1, 2)) > 0]

        points_coords, points_labels = generate_random_points(masks)
        data['points_coords'] = points_coords
        data['points_labels'] = points_labels
        data['masks'] = masks
        data['image'] = self.image_transform(data['image'])

        return data
