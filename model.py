from lightning import LightningModule
from sam2_modified import SAM2ImagePredictorTensor
from sam2.build_sam import build_sam2
import torch


class SAM2Model(LightningModule):
    def __init__(self,
                 checkpoint_path: str = "sam2-checkpoints/sam2_hiera_small.pt",
                 model_cfg: str = "sam2_hiera_s.yaml",
                 device: str = "cuda"):
        super().__init__()
        sam2_checkpoint = checkpoint_path
        model_cfg = model_cfg
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictorTensor(sam2_model)
        # register self.predictor.model as a submodule
        self.add_module("sam2_model", self.predictor.model)

    def freeze_all(self,
                   freeze_mask_decoder=True,
                   freeze_prompt_encoder=True,
                   freeze_adapter=False):
        for _, param in self.predictor.model.named_parameters():
            param.requires_grad = False
        if not freeze_mask_decoder:
            for _, param in self.predictor.model.sam_mask_decoder.named_parameters():
                param.requires_grad = True
        if not freeze_prompt_encoder:
            for _, param in self.predictor.model.sam_prompt_encoder.named_parameters():
                param.requires_grad = True
        if not freeze_adapter:
            for name, param in self.predictor.model.named_parameters():
                if 'adapter' in name:
                    param.requires_grad = True

    def forward(self, x):
        return self.predictor.model.forward_image(x)

    def _forward_step(self,
                      images: torch.Tensor,
                      masks,
                      boxes,
                      input_points,
                      input_labels) -> tuple[float, float]:
        """
        images.shape: (B, C, H, W)
        masks.shape: (B, N_i, H, W)
        input_points.shape: (B, 1, N_i, 2)
        input_labels.shape: (B, 1, N_i)
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
            mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(in_point,
                                                                                         in_label,
                                                                                         box=in_box,
                                                                                         mask_logits=None,
                                                                                         normalize_coords=True,
                                                                                         img_idx=img_idx)
            concat_points = (unnorm_coords, labels) if in_point is not None else None
            sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=mask_input,
            )

            # mask decoder

            batched_mode = concat_points is not None and concat_points[0].shape[0] > 1  # multi object prediction
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
                                      lr=1e-4,
                                      weight_decay=1e-5)
        return optimizer

    def compute_losses_old(self, mask, prd_masks, prd_scores) -> tuple[torch.Tensor, torch.Tensor]:
        # Segmentation Loss caclulation

        prd_scores, pred_max_idx = prd_scores.max(1)
        # There are 3 outputs in the mask decoder for ambiguity resolution

        # prd_masks will have the same indices selected by pred_max_idx
        prd_masks = prd_masks[torch.arange(prd_masks.shape[0]), pred_max_idx]

        gt_mask = mask.to(dtype=torch.float32)
        prd_mask = torch.sigmoid(prd_masks)
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask)
                    * torch.log((1 - prd_mask) + 0.00001)).mean()

        # Score loss calculation (intersection over union) IOU

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores - iou).mean()
        loss = seg_loss+score_loss*0.05  # mix losses

        return loss, iou.mean()

    def compute_losses(self, mask, prd_masks, prd_scores) -> tuple[torch.Tensor, torch.Tensor]:
        # # Segmentation Loss caclulation

        gt_mask = mask.to(dtype=torch.float32)
        prd_mask = torch.sigmoid(prd_masks)
        gt_mask_aug = gt_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        seg_loss_pointwise = (-gt_mask_aug * torch.log(prd_mask + 0.00001) - (1 - gt_mask_aug)
                              * torch.log((1 - prd_mask) + 0.00001))

        seg_loss, min_idx = seg_loss_pointwise.mean((0, 2, 3)).min(0)

        prd_mask = prd_mask[:, min_idx]
        prd_scores = prd_scores[:, min_idx]

        # Score loss calculation (intersection over union) IOU

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores - iou).mean()
        loss = seg_loss+score_loss*0.05  # mix losses

        return loss, iou.mean()
