import torch
from torch.utils.data import Dataset
from typing import Any
import numpy as np
from torchvision import transforms as T
from sam2_modified import SAM2TransformsTensor


def generate_random_points(masks: torch.Tensor,
                           num_points: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    if len(masks) == 0:
        masks = torch.zeros(size=(0,))
        points = torch.zeros(size=(0,))
    else:
        points = []
        for mask in masks:
            coords = torch.argwhere(mask > 0)
            if len(coords) == 0:
                continue
            ps = []
            for i in range(num_points):
                yx = coords[np.random.randint(len(coords))]
                ps.append([yx[1], yx[0]])
            points.append(ps)
        points = torch.tensor(points)

    return points, torch.ones((len(points), len(points[0])))


def generate_box(masks: torch.Tensor) -> torch.Tensor:
    if len(masks) == 0:
        return torch.zeros(size=(0,))
    else:
        boxes = []
        for mask in masks:
            coords = torch.argwhere(mask > 0)
            if len(coords) == 0:
                continue
            y1, x1 = coords.min(0)[0]
            y2, x2 = coords.max(0)[0]
            boxes.append([x1, y1, x2, y2])
        return torch.tensor(boxes)


def split_masks(masks: torch.Tensor) -> torch.Tensor:
    """
    For each unique value in the mask, create a new mask.
    From (H,W) to (N, H, W).
    """
    unique_vals = torch.unique(masks)
    unique_vals = unique_vals[unique_vals > 0]
    return torch.stack([masks == val for val in unique_vals]).to(dtype=torch.uint8)


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
        # assert isinstance(image_transform, SAM2TransformsTensor)
        # self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> dict[str, Any]:
        data: dict = self.dataset[index]
        masks = data['masks']
        if masks.shape[0] != 0:
            masks = self.mask_transform(masks)
            masks = masks[masks.sum((1, 2)) > 0]

        img = data['image']

        # min-max normalization
        if torch.is_tensor(img):
            img = img.to(dtype=torch.float32)
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())

        boxes = None
        points_coords, points_labels = generate_random_points(masks,
                                                              num_points=np.random.randint(1, 4) # 1 to 3 points
                                                              )
        data['points_coords'] = points_coords
        data['points_labels'] = points_labels
        data['boxes'] = boxes
        data['masks'] = masks
        data['image'] = self.mask_transform(img)

        return data
