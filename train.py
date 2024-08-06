
# Train/Fine Tune SAM 2 on LabPics 1 dataset
# Labpics can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from torch.utils.data import DataLoader, Dataset
import lightning as L
import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm.auto import tqdm
from torchvision import transforms as T


class LabPicsDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 folder_name: str = 'Train',
                 transform=None):
        self.data = self._scan_data(data_dir, folder_name)
        self.mask_transform = T.Compose([
            T.ToTensor(),
            T.Resize((1024, 1024), interpolation=T.InterpolationMode.NEAREST, antialias=False)
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
    def _read_single_image(self, ent: dict) -> tuple:
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
        return img, masks, points, torch.ones((len(masks), 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        img, masks, points, labels = self._read_single_image(self.data[idx])
        return img, masks, points, labels


class LabPicsDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage: str = None):
        self.train_dataset = LabPicsDataset(self.data_dir, 'Train', transform=self.transform)
        # self.val_dataset = LabPicsDataset(self.data_dir, 'Test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4
                          )

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size)


class SAM2Model(LightningModule):
    def __init__(self,
                 checkpoint_path: str = "sam2-checkpoints/sam2_hiera_small.pt",
                 model_cfg: str = "sam2_hiera_s.yaml",
                 device: str = "cuda"):
        super().__init__()
        sam2_checkpoint = checkpoint_path
        model_cfg = model_cfg
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

        # Set training parameters

        self.predictor.model.sam_mask_decoder.train(True)
        self.predictor.model.sam_prompt_encoder.train(True)

    def forward(self, x):
        return self.predictor.model.forward_image(x)

    def _forward_step(self,
                      image: torch.Tensor,
                      mask,
                      input_point,
                      input_label) -> tuple[torch.Tensor, torch.Tensor]:
        """
        image.shape: (B, C, H, W)
        mask.shape: (B, N_i, H, W)
        input_point.shape: (B, N_i, 1, 2)
        input_label.shape: (B, N_i, 1)
        """

        # prompt encoding
        if mask.shape[1] == 0:
            return torch.tensor(0.0, requires_grad=True), torch.tensor(0.0)
        image = image[0]
        mask = mask[0]
        input_point = input_point[0]
        input_label = input_label[0]
        self.predictor.set_image(image)
        mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
        sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None,)

        # mask decoder

        batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in self.predictor._features["high_res_feats"]]
        image_pe = self.predictor.model.sam_prompt_encoder.get_dense_pe()
        low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),
                                                                                image_pe=image_pe,
                                                                                sparse_prompt_embeddings=sparse_embeddings,
                                                                                dense_prompt_embeddings=dense_embeddings,
                                                                                multimask_output=True,
                                                                                repeat_image=batched_mode,
                                                                                high_res_features=high_res_features)
        # Upscale the masks to the original image resolution
        prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])

        return self.compute_losses(mask, prd_masks, prd_scores)

    def training_step(self, batch, batch_idx):
        image, mask, input_point, input_label = batch
        # image.shape: (B, C, H, W)
        loss, iou = self._forward_step(image, mask, input_point, input_label)
        self.log('train_loss', loss)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     images, annotations = batch
    #     predictions = self(images)
    #     loss = self.compute_loss(predictions, annotations)
    #     self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.predictor.model.parameters(),
                                      lr=1e-5,
                                      weight_decay=4e-5)
        return optimizer

    def compute_losses(self, mask, prd_masks, prd_scores) -> tuple[torch.Tensor, torch.Tensor]:
        # Segmentation Loss caclulation

        gt_mask = mask.to(torch.float32)
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask)
                    * torch.log((1 - prd_mask) + 0.00001)).mean()

        # Score loss calculation (intersection over union) IOU

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss+score_loss*0.05  # mix losses

        return loss, iou


# Main script
if __name__ == "__main__":
    data_dir = "data/LabPicsV1/"  # Path to dataset (LabPics 1)
    batch_size = 1

    model = SAM2Model()
    data_module = LabPicsDataModule(data_dir, batch_size,
                                    transform=model.predictor._transforms)

    # checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')

    trainer = Trainer(max_epochs=3,
                      precision="bf16",
                      #   callbacks=[checkpoint_callback]
                      )
    trainer.fit(model, data_module)
