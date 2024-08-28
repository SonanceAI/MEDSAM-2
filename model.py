from lightning import LightningModule
from sam2_modified import SAM2ImagePredictorTensor
from sam2.build_sam import build_sam2
import torch
import logging
from collections import defaultdict
from torch.nn import BCEWithLogitsLoss

_LOGGER = logging.getLogger(__name__)


class SAM2Model(LightningModule):
    def __init__(self,
                 checkpoint_path: str = "sam2-checkpoints/sam2_hiera_small.pt",
                 model_cfg: str = "sam2_hiera_s.yaml",
                 device: str = "cuda",
                 learning_rate: float = 1e-5
                 ):
        super().__init__()
        self.learning_rate = learning_rate
        sam2_checkpoint = checkpoint_path
        model_cfg = model_cfg
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictorTensor(sam2_model)
        # register self.predictor.model as a submodule
        self.add_module("sam2_model", self.predictor.model)
        self.mask_criterion = BCEWithLogitsLoss(reduction='none')
        self.train(True)
        self.predictor.model.train(True)

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
                      input_labels) -> dict[str, torch.Tensor]:
        """
        images.shape: (B, C, H, W)
        masks.shape: (B, N_i, H, W)
        input_points.shape: (B, 1, N_i, 2)
        input_labels.shape: (B, 1, N_i)
        """
        self.predictor.set_image_batch(images)
        metrics = defaultdict(float)
        k = 0
        for img_idx in range(len(images)):
            in_point = input_points[img_idx] if input_points is not None else None
            in_label = input_labels[img_idx] if input_labels is not None else None
            in_box = boxes[img_idx] if boxes is not None else None
            mask = masks[img_idx]
            if mask.shape[0] == 0:
                _LOGGER.warning(f"Skipping image {img_idx} as it has no masks")
                continue
            mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(in_point,
                                                                                         in_label,
                                                                                         box=in_box,
                                                                                         mask_logits=None,
                                                                                         normalize_coords=True,
                                                                                         img_idx=img_idx)
            if in_point is not None:
                concat_points = (unnorm_coords, labels)
                batched_mode = unnorm_coords.shape[0] > 1
            else:
                concat_points = None
                batched_mode = False
            sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=mask_input,
            )

            # mask decoder
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

            metrics_i = self.compute_losses(mask, prd_masks, prd_scores)
            for key, value in metrics_i.items():
                metrics[key] += value
            k += 1
        return {key: v/k for key, v in metrics.items()}

    def training_step(self, batch: dict, batch_idx):
        image = batch['image']
        mask = batch['masks']
        input_point = batch.get('points_coords', None)
        input_label = batch.get('points_labels', None)
        boxes = batch.get('boxes', None)

        batch_size = len(image)
        # image.shape: (B, C, H, W)
        metrics_dict = self._forward_step(image,
                                          masks=mask,
                                          boxes=boxes,
                                          input_points=input_point,
                                          input_labels=input_label)
        loss = metrics_dict['loss']
        self.log('train/loss', loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        for key, value in metrics_dict.items():
            if key != 'loss':
                self.log(f'train/{key}', value, on_epoch=True, on_step=False, batch_size=batch_size)
        return loss

    def validation_step(self, batch: dict, batch_idx):
        image = batch['image']
        mask = batch['masks']
        input_point = batch['points_coords']
        input_label = batch['points_labels']
        boxes = batch['boxes']

        batch_size = len(image)
        # image.shape: (B, C, H, W)
        metrics_dict = self._forward_step(image,
                                          masks=mask,
                                          boxes=boxes,
                                          input_points=input_point,
                                          input_labels=input_label)
        loss = metrics_dict['loss']
        self.log('val/loss', loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        for key, value in metrics_dict.items():
            if key != 'loss':
                self.log(f'val/{key}', value, on_epoch=True, on_step=False, batch_size=batch_size)

    def configure_optimizers(self):
        adapter_params = [param for name, param in self.predictor.model.named_parameters() if 'adapter' in name]

        params_to_optim = [
            {'params': self.predictor.model.sam_mask_decoder.parameters(), 'lr': self.learning_rate /
             10, 'weight_decay': 1e-4},
            {'params': adapter_params, 'lr': self.learning_rate, 'weight_decay': 1e-5}
        ]
        optimizer = torch.optim.AdamW(params_to_optim,
                                      lr=self.learning_rate/10,
                                      weight_decay=0)
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

    def compute_losses(self, mask, prd_masks, prd_scores) -> dict[str, torch.Tensor]:
        # Segmentation Loss calculation

        gt_mask = mask.to(dtype=torch.float32)
        gt_mask_aug = gt_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        seg_loss_pointwise = self.mask_criterion(prd_masks, gt_mask_aug)

        seg_loss, min_idx = seg_loss_pointwise.mean((0, 2, 3)).min(0)

        prd_masks = prd_masks[:, min_idx]
        prd_scores = prd_scores[:, min_idx]

        # Score loss calculation (intersection over union) IOU
        prd_01 = prd_masks > 0.0
        inter = (gt_mask * prd_01).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + prd_01.sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores - iou).mean()
        loss = seg_loss+score_loss*0.05  # mix losses

        return {'loss': loss, 'seg_loss': seg_loss, 'score_loss': score_loss, 'iou': iou.mean()}
