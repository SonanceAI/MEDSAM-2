
# Train/Fine Tune SAM 2 on LabPics 1 dataset
# Labpics can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning import Trainer
import torch
from torch.utils.data import DataLoader
from datasets.sam_dataset import SAMDataset, collate_dict_SAM
from datasets import CAMUS, USForKidney, BreastUS, USsimandsegmDataset, USnervesegDataset, USThyroidDataset
import os
from torch.utils.data import ConcatDataset
import logging
import logging.config
from model import SAM2Model

_LOGGER = logging.getLogger(__name__)


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

    ### BreastUS ###
    breastus_dir = os.path.join(root_dir, 'breast-ultrasound-images-dataset')
    train_dataset_list.append(SAMDataset(BreastUS(breastus_dir, 'train'),
                                         image_transform=transforms)
                              )
    val_dataset_list.append(SAMDataset(BreastUS(breastus_dir, 'test'),
                                       image_transform=transforms)
                            )

    ### USsimandsegm ###
    ussimandsegm_dir = os.path.join(root_dir, 'ussimandsegm')
    train_dataset_list.append(SAMDataset(USsimandsegmDataset(ussimandsegm_dir, 'train'),
                                         image_transform=transforms)
                              )
    val_dataset_list.append(SAMDataset(USsimandsegmDataset(ussimandsegm_dir, 'test'),
                                       image_transform=transforms)
                            )

    ### USnerveseg ###
    usnerveseg_dir = os.path.join(root_dir, 'ultrasound-nerve-segmentation')
    train_dataset_list.append(SAMDataset(USnervesegDataset(usnerveseg_dir),
                                         image_transform=transforms)
                              )
    val_dataset_list.append(SAMDataset(USnervesegDataset(usnerveseg_dir),
                                       image_transform=transforms)
                            )

    ### USThyroidDataset ###
    usthyroid_dir = os.path.join(root_dir, 'Thyroid Dataset')
    train_dataset_list.append(SAMDataset(USThyroidDataset(usthyroid_dir, 'train'),
                                         image_transform=transforms)
                              )
    val_dataset_list.append(SAMDataset(USThyroidDataset(usthyroid_dir, 'test'),
                                       image_transform=transforms)
                            )

    return train_dataset_list, val_dataset_list


def main():
    torch.set_float32_matmul_precision('high')
    root_dir = "data/raw"
    batch_size = 8

    model = SAM2Model(checkpoint_path="sam2-checkpoints/sam2_hiera_small.pt",
                      model_cfg="sam2_hiera_s.yaml")
    model.freeze_all(freeze_mask_decoder=False)
    train_dataset_list, val_dataset_list = load_datasets(root_dir, model.predictor._transforms)
    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)

    _LOGGER.info(f"Train dataset size: {len(train_dataset)}")
    _LOGGER.info(f"Validation dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=10,
                                  pin_memory=True,
                                  persistent_workers=True,
                                  collate_fn=collate_dict_SAM)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=10,
                                shuffle=False,
                                pin_memory=True,
                                persistent_workers=True,
                                collate_fn=collate_dict_SAM)

    checkpoint_callback = ModelCheckpoint(monitor='val/iou',
                                          filename='sam2-{epoch:02d}-{val/loss:.2f}',
                                          dirpath='checkpoints',
                                          mode='max')

    trainer = Trainer(max_epochs=30,
                      num_sanity_val_steps=0,
                      accelerator='cuda',
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
