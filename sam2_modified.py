from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms
import torch
from typing import Union, Sequence
import numpy as np
from PIL.Image import Image
import logging
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2TransformsTensor(SAM2Transforms):
    def __call__(self, x):
        if not torch.is_tensor(x):
            x = self.to_tensor(x)
        return self.transforms(x)

    def forward_batch(self, img_list):
        img_batch = [self.__call__(img) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch


class SAM2ImagePredictorTensor(SAM2ImagePredictor):
    def __init__(self, sam_model: SAM2Base, mask_threshold=0, max_hole_area=0, max_sprinkle_area=0) -> None:
        super().__init__(sam_model, mask_threshold, max_hole_area, max_sprinkle_area)
        self._transforms = SAM2TransformsTensor(resolution=self.model.image_size,
                                                mask_threshold=mask_threshold,
                                                max_hole_area=max_hole_area,
                                                max_sprinkle_area=max_sprinkle_area)

    def set_image(
        self,
        image: Union[np.ndarray, Image, torch.Tensor],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, torch.Tensor):
            logging.info("For torch tensor image, we assume (CxHxW) format")
            self._orig_hw = [(image.shape[1], image.shape[2])]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    def set_image_batch(
        self,
        image_list: Sequence[Union[np.ndarray, torch.Tensor]],
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.

        Arguments:
          image_list (List[np.ndarray]): The input images to embed in RGB format. The image should be in HWC format if np.ndarray
          with pixel values in [0, 255].
        """
        self.reset_predictor()
        # assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            # assert isinstance(
            #     image, np.ndarray
            # ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
            # self._orig_hw.append(image.shape[:2])
            if isinstance(image, np.ndarray):
                self._orig_hw.append(image.shape[:2])
            elif isinstance(image, torch.Tensor):
                self._orig_hw.append((image.shape[1], image.shape[2]))
            else:
                raise NotImplementedError("Image format not supported")
        # Transform the image to the form expected by the model
        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")
