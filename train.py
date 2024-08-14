
# Train/Fine Tune SAM 2 on LabPics 1 dataset
# Labpics can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning import LightningModule, Trainer
import torch
from sam2.build_sam import build_sam2
from sam2_modified import SAM2ImagePredictorTensor
from datasets.labpics import LabPicsDataModule, LabPicsDataset
from torch.utils.data import DataLoader
from datasets.sam_dataset import SAMDataset, collate_dict_SAM
from datasets import CAMUS, USForKidney
import os
from torch.utils.data import ConcatDataset
import logging
import logging.config

_LOGGER = logging.getLogger(__name__)


class SAM2Model(LightningModule):
    def __init__(self,
                 checkpoint_path: str = "sam2-checkpoints/sam2_hiera_small.pt",
                 model_cfg: str = "sam2_hiera_s.yaml",
                 device: str = "cuda"):
        super().__init__()
        sam2_checkpoint = checkpoint_path
        model_cfg = model_cfg
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictorTensor(self.sam2_model)
        # Set training parameters

        self.predictor.model.sam_mask_decoder.train(True)
        self.predictor.model.sam_prompt_encoder.train(True)

    def forward(self, x):
        return self.predictor.model.forward_image(x)

    def _forward_step(self,
                      images: torch.Tensor,
                      masks,
                      boxes,
                      input_points,
                      input_labels) -> tuple[torch.Tensor, torch.Tensor]:
        """
        images.shape: (B, C, H, W)
        masks.shape: (B, N_i, H, W)
        input_points.shape: (B, N_i, 1, 2)
        input_labels.shape: (B, N_i, 1)
        """

        self.predictor.set_image_batch(images)
        losses = []
        ious = []
        for img_idx in range(len(images)):
            in_point = input_points[img_idx] if input_points is not None else None
            in_label = input_labels[img_idx] if input_labels is not None else None
            in_box = boxes[img_idx] if boxes is not None else None
            mask = masks[img_idx]
            if mask.shape[0] == 0:
                continue
            _, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(in_point,
                                                                                in_label,
                                                                                box=in_box,
                                                                                mask_logits=None,
                                                                                normalize_coords=True,
                                                                                img_idx=img_idx)
            prep_points = (unnorm_coords, labels) if in_point is not None else None
            sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                points=prep_points,
                boxes=unnorm_box,
                masks=None,
            )

            # mask decoder

            batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
            high_res_features = [feat_level[img_idx].unsqueeze(0)
                                 for feat_level in self.predictor._features["high_res_feats"]]
            image_pe = self.predictor.model.sam_prompt_encoder.get_dense_pe()
            low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(image_embeddings=self.predictor._features["image_embed"][img_idx].unsqueeze(0),
                                                                                    image_pe=image_pe,
                                                                                    sparse_prompt_embeddings=sparse_embeddings,
                                                                                    dense_prompt_embeddings=dense_embeddings,
                                                                                    multimask_output=True,
                                                                                    repeat_image=batched_mode,
                                                                                    high_res_features=high_res_features)
            # Upscale the masks to the original image resolution
            prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[img_idx])
            # low_res_masks.shape: (N_i, 3, 256, 256). prd_scores.shape: (N_i, 3). prd_masks.shape: (N_i, H, W)

            loss, iou = self.compute_losses(mask, prd_masks, prd_scores)
            losses.append(loss)
            ious.append(iou)
        return torch.stack(losses).mean(), torch.stack(ious).mean()  # check what is the best, 'mean' or 'sum'.

    def training_step(self, batch: dict, batch_idx):
        image = batch['image']
        mask = batch['masks']
        input_point = batch.get('points_coords', None)
        input_label = batch.get('points_labels', None)
        boxes = batch.get('boxes', None)

        batch_size = len(image)
        # image.shape: (B, C, H, W)
        loss, iou = self._forward_step(image,
                                       masks=mask,
                                       boxes=boxes,
                                       input_points=input_point,
                                       input_labels=input_label)
        self.log('train/loss', loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log('train/iou', iou, on_epoch=True, on_step=False, batch_size=batch_size)
        return loss

    def validation_step(self, batch: dict, batch_idx):
        image = batch['image']
        mask = batch['masks']
        input_point = batch['points_coords']
        input_label = batch['points_labels']
        boxes = batch['boxes']

        batch_size = len(image)
        # image.shape: (B, C, H, W)
        loss, iou = self._forward_step(image,
                                       masks=mask,
                                       boxes=boxes,
                                       input_points=input_point,
                                       input_labels=input_label)
        self.log('val/loss', loss, prog_bar=True, batch_size=batch_size)
        self.log('val/iou', iou, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)

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

        return loss, iou.mean()


def load_datasets(root_dir: str, transforms) -> tuple[list, list]:
    train_dataset_list = []
    val_dataset_list = []

    ### CAMUS ###
    camus_dir = os.path.join(root_dir, 'CAMUS')
    train_dataset_list.append(SAMDataset(CAMUS(camus_dir, 'train'),
                                         image_transform=transforms)
                              )
    val_dataset_list.append(SAMDataset(CAMUS(camus_dir, 'test'),
                                       image_transform=transforms)
                            )

    ### ct2usforkidneyseg ###
    usforkidney_dir = os.path.join(root_dir, 'ct2usforkidneyseg')
    train_dataset_list.append(SAMDataset(USForKidney(usforkidney_dir, 'train'),
                                         image_transform=transforms)
                              )
    val_dataset_list.append(SAMDataset(USForKidney(usforkidney_dir, 'test'),
                                       image_transform=transforms)
                            )

    return train_dataset_list, val_dataset_list


def main():
    torch.set_float32_matmul_precision('high')
    root_dir = "data/raw"
    batch_size = 4

    model = SAM2Model()
    train_dataset_list, val_dataset_list = load_datasets(root_dir, model.predictor._transforms)
    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)

    _LOGGER.info(f"Train dataset size: {len(train_dataset)}")
    _LOGGER.info(f"Validation dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=6,
                                  collate_fn=collate_dict_SAM)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=6,
                                collate_fn=collate_dict_SAM)

    checkpoint_callback = ModelCheckpoint(monitor='val/iou',
                                          filename='sam2-{epoch:02d}-{val/loss:.2f}',
                                          dirpath='checkpoints',
                                          mode='max')

    trainer = Trainer(max_epochs=10,
                      precision="bf16-mixed",
                      callbacks=[checkpoint_callback, RichModelSummary()],
                      )
    trainer.validate(model, val_dataloader)
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
                )


# Main script
if __name__ == "__main__":
    from rich.logging import RichHandler

    logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)],
                        format="%(message)s")
    logging.getLogger(__name__).setLevel(logging.INFO)

    main()
