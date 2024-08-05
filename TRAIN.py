
# Train/Fine Tune SAM 2 on LabPics 1 dataset
# Labpics can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1

import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm.auto import tqdm

NUM_ITERS = 100000

# Read data


def read_data(data_dir: str, folder_name: str = 'Train') -> list[dict]:
    data = []
    for ff, name in enumerate(os.listdir(f"{data_dir}/Simple/{folder_name}/Image/")):
        data.append({"image": f"{data_dir}/Simple/{folder_name}/Image/"+name,
                    "annotation": f"{data_dir}/Simple/{folder_name}/Instance/"+name[:-4]+".png"})
    return data


data_dir = r"data/LabPicsV1/"  # Path to dataset (LabPics 1)
train_data = read_data(data_dir)
val_data = read_data(data_dir, 'Test')

print(f'Number of training images in the dataset: {len(train_data)}')


# read random image and its annotaion from  the dataset (LabPics)
def read_batch(data: list[dict], idx: int | None = None) -> tuple:

   #  select image
    if idx is None:
        ent = data[np.random.randint(len(data))]
    else:
        ent = data[idx]
    Img = cv2.imread(ent["image"])[..., ::-1]  # read image
    ann_map = cv2.imread(ent["annotation"])

   # resize image

    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r),
                         int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

   # merge vessels and materials annotations

    mat_map = ann_map[:, :, 0]
    ves_map = ann_map[:, :, 2]
    mat_map[mat_map == 0] = ves_map[mat_map == 0]*(mat_map.max()+1)

   # Get binary masks and points

    inds = np.unique(mat_map)[1:]
    points = []
    masks = []
    for ind in inds:
        mask = (mat_map == ind).astype(np.uint8)
        masks.append(mask)
        coords = np.argwhere(mask > 0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return Img, np.array(masks), np.array(points), np.ones([len(masks), 1])

# Load model


sam2_checkpoint = "sam2-checkpoints/sam2_hiera_small.pt"  # "sam2_hiera_large.pt"
model_cfg = "sam2_hiera_s.yaml"  # "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters

predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)
optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler()  # mixed precision


def compute_losses(mask, prd_masks, prd_scores) -> tuple:
    # Segmentation Loss caclulation

    gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
    prd_mask = torch.sigmoid(prd_masks[:, 0])
    seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask)
                * torch.log((1 - prd_mask) + 0.00001)).mean()

    # Score loss calculation (intersection over union) IOU

    inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
    iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
    loss = seg_loss+score_loss*0.05  # mix losses

    return loss, iou


def forward_step(image, mask, input_point, input_label) -> tuple:
    # prompt encoding

    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
        input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=(unnorm_coords, labels), boxes=None, masks=None,)

    # mask decoder

    batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0), image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(
    ), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=True, repeat_image=batched_mode, high_res_features=high_res_features,)
    # Upscale the masks to the original image resolution
    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

    return compute_losses(mask, prd_masks, prd_scores)


def validate(val_data) -> float:
    with torch.no_grad():
        ious = []
        for i in tqdm(range(len(val_data))):
            image, mask, input_point, input_label = read_batch(val_data, idx=i)
            predictor.set_image(image)
            loss, iou = forward_step(image, mask, input_point, input_label)
            ious.append(iou.cpu().numpy().mean())
        mean_iou = np.mean(ious)
        return mean_iou


mean_iou = 0
# Training loop
with tqdm(total=NUM_ITERS) as pbar:
    for itr in range(NUM_ITERS):
        with torch.amp.autocast('cuda'):  # cast to mix precision
            # with torch.cuda.amp.autocast():
            image, mask, input_point, input_label = read_batch(train_data)  # load data batch
            if mask.shape[0] == 0:
                continue  # ignore empty batches
            predictor.set_image(image)  # apply SAM image encoder to the image

            loss, iou = forward_step(image, mask, input_point, input_label)

            # apply back propogation

            predictor.model.zero_grad()  # empty gradient
            scaler.scale(loss).backward()  # Backpropogate
            scaler.step(optimizer)
            scaler.update()  # Mix precision
            # Display results

            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            if itr != 0 and itr % 25 == 0:
                pbar.update(25)
                pbar.set_description(f"Train accuracy(IOU)={mean_iou:.2%}")

            if itr % 1000 == 0:
                # Validation
                val_mean_iou = validate(val_data)
                print(f"step) {itr:05d} Val.Acc(IOU)={val_mean_iou:.2%}")
                torch.save(predictor.model.state_dict(), f"checkpoints/model-{itr:05}-{val_mean_iou:.4f}.pth")  # save model
