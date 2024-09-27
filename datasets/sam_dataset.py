import torch
from torch.utils.data import Dataset
from typing import Any
import numpy as np
from torchvision import transforms as T
from sam2_modified import SAM2TransformsTensor
import logging

_LOGGER = logging.getLogger(__name__)


def generate_random_points(masks: torch.Tensor,
                           num_pos_points: int = 1,
                           num_neg_points: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if len(masks) == 0:
        masks = torch.zeros(size=(0,))
        points_pos = torch.zeros(size=(0,))
        points_neg = torch.zeros(size=(0,))
    else:
        points_pos = []
        for mask in masks:
            coords = torch.argwhere(mask > 0)
            if len(coords) == 0:
                continue
            if len(coords) < 8:
                _LOGGER.warning(f"Very few points in mask: {len(coords)}")
                # raise Exception(f"Very few points in mask: {len(coords)}")
            num_pos_points = min(num_pos_points, len(coords))
            # choose num_pos_points random points from coords
            idx = np.random.choice(len(coords), num_pos_points, replace=False)
            points_pos.append(coords[idx])
        points_pos = torch.stack(points_pos)

        if num_neg_points > 0:
            points_neg = []
            for mask in masks:
                coords = torch.argwhere(mask == 0)
                if len(coords) == 0:
                    continue
                # choose num_pos_points random points from coords
                idx = np.random.choice(len(coords), num_neg_points, replace=False)
                points_neg.append(coords[idx])
            points_neg = torch.stack(points_neg)
        else:
            points_neg = torch.zeros(size=(len(masks), 0, 2), dtype=torch.int)

    points = torch.cat((points_pos, points_neg), dim=1)  # shape: (#masks, num_pos_points+num_neg_points, 2)

    # swap the values of the last columns
    x = points[:, :, 0].clone()
    points[:, :, 0] = points[:, :, 1]
    points[:, :, 1] = x
    labels = torch.ones((len(points_pos), num_pos_points+num_neg_points), dtype=torch.int)
    labels[:, num_pos_points:] = 0
    return points, labels


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


def split_masks(masks: torch.Tensor,
                black_color: int = 0) -> torch.Tensor:
    """
    For each unique value in the mask, create a new mask.
    From (H,W) or (H,W,3) to (N, H, W).
    """

    if masks.ndim == 3:
        assert masks.shape[2] == 3
        # compute all unique colors
        unique_vals = torch.unique(masks.reshape(-1, 3), dim=0)
        # discard black color
        unique_vals = unique_vals[unique_vals.sum(dim=1) > black_color]
        if len(unique_vals) == 0:
            return torch.zeros(size=(0,))
        return torch.stack([torch.all(masks == val, dim=-1) for val in unique_vals]).to(dtype=torch.uint8)
    else:
        unique_vals = torch.unique(masks)
        # discard black color
        unique_vals = unique_vals[unique_vals > black_color]
        if len(unique_vals) == 0:
            return torch.zeros(size=(0,))
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
                 image_transform: SAM2TransformsTensor = None) -> None:
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

        if len(masks) > 60:
            _LOGGER.warning(f"Too many masks: {len(masks)}! Check the dataset.")

        img = data['image']
        if np.random.randint(2) == 1:
            # rotate 90 degrees
            masks = masks.permute(0, 2, 1)
            img = img.permute(0, 2, 1)

        if np.random.randint(2) == 1:
            # flip horizontally
            masks = masks.flip(-1)
            img = img.flip(-1)

        # min-max normalization
        if torch.is_tensor(img):
            img = img.to(dtype=torch.float32)
        else:
            img = img.astype(np.float32)
        mn = img.min()
        img = (img - mn) / (img.max() - mn)

        boxes = None
        try:
            points_coords, points_labels = generate_random_points(masks,
                                                                  num_pos_points=np.random.randint(
                                                                      1, 3),  # 1 to 2 points
                                                                  num_neg_points=np.random.randint(
                                                                      0, 2)  # 0 to 1 points
                                                                  )
        except Exception as e:
            _LOGGER.error(f"Error in generate_random_points [{data['image_name']}]: {e}")
            raise e
        data['points_coords'] = points_coords
        data['points_labels'] = points_labels
        data['boxes'] = boxes
        data['masks'] = masks
        data['image'] = self.mask_transform(img)

        return data
