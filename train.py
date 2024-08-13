
# Train/Fine Tune SAM 2 on LabPics 1 dataset
# Labpics can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning import LightningModule, Trainer
import torch
from sam2.build_sam import build_sam2
from sam2_modified import SAM2ImagePredictorTensor
from datasets.labpics import LabPicsDataModule


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
            in_point = input_points[img_idx]
            in_label = input_labels[img_idx]
            mask = masks[img_idx]
            if mask.shape[0] == 0:
                continue
            _, unnorm_coords, labels, _ = self.predictor._prep_prompts(in_point,
                                                                       in_label,
                                                                       box=None,
                                                                       mask_logits=None,
                                                                       normalize_coords=True,
                                                                       img_idx=img_idx)
            sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels),
                boxes=None,
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
        input_point = batch['points_coords']
        input_label = batch['points_labels']

        batch_size = len(image)
        # image.shape: (B, C, H, W)
        loss, iou = self._forward_step(image, mask, input_point, input_label)
        self.log('train/loss', loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log('train/iou', iou, on_epoch=True, on_step=False, batch_size=batch_size)
        return loss

    def validation_step(self, batch: dict, batch_idx):
        image = batch['image']
        mask = batch['masks']
        input_point = batch['points_coords']
        input_label = batch['points_labels']

        batch_size = len(image)
        # image.shape: (B, C, H, W)
        loss, iou = self._forward_step(image, mask, input_point, input_label)
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


# Main script
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    data_dir = "data/LabPicsV1/"
    batch_size = 8

    model = SAM2Model()
    data_module = LabPicsDataModule(data_dir, batch_size,
                                    transform=model.predictor._transforms)

    checkpoint_callback = ModelCheckpoint(monitor='val/iou',
                                          filename='sam2-{epoch:02d}-{val/loss:.2f}',
                                          dirpath='checkpoints',
                                          mode='min')

    trainer = Trainer(max_epochs=3,
                      precision="bf16-mixed",
                      callbacks=[checkpoint_callback, RichModelSummary()],
                      )
    trainer.fit(model, data_module)
