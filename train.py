
# Train/Fine Tune SAM 2 on LabPics 1 dataset
# Labpics can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning import Trainer
import torch
from torch.utils.data import DataLoader, ConcatDataset
from datasets.sam_dataset import SAMDataset, collate_dict_SAM
from datasets import (CAMUS, USForKidney, BreastUS, USsimandsegmDataset, USnervesegDataset,
                      USThyroidDataset, FHPSAOPDataset, BUS_UC, BUS_UCLM, BUSCv, FetalAbnominal,
                      BUSI_WHU, RotatorCuffDataset)
import os
import logging
import logging.config
from model import SAM2Model
from lightning.pytorch.loggers import TensorBoardLogger
import datetime

# For ensuring reproducibility
# torch.manual_seed(12)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


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

    ### USThyroidDataset ###
    usthyroid_dir = os.path.join(root_dir, 'Thyroid Dataset')
    train_dataset_list.append(SAMDataset(USThyroidDataset(usthyroid_dir, 'train'),
                                         image_transform=transforms)
                              )
    val_dataset_list.append(SAMDataset(USThyroidDataset(usthyroid_dir, 'test'),
                                       image_transform=transforms)
                            )

    ### FHPSAOPDataset ###
    fhpsaop_dir = os.path.join(root_dir, 'FH-PS-AOP')
    train_dataset_list.append(SAMDataset(FHPSAOPDataset(fhpsaop_dir, 'train'),
                                         image_transform=transforms)
                              )
    val_dataset_list.append(SAMDataset(FHPSAOPDataset(fhpsaop_dir, 'test'),
                                       image_transform=transforms)
                            )

    ### BUS_UC ###
    busuc_dir = os.path.join(root_dir, 'BUS_UC')
    train_dataset_list.append(SAMDataset(BUS_UC(busuc_dir, 'train'),
                                         image_transform=transforms)
                              )
    val_dataset_list.append(SAMDataset(BUS_UC(busuc_dir, 'test'),
                                       image_transform=transforms)
                            )

    return train_dataset_list, val_dataset_list


def load_datasets2(root_dir: str, transforms) -> tuple[list, list]:
    """
    Load datasets separating a single complete dataset to test.
    """
    train_dataset_list = []
    val_dataset_list = []

    ### CAMUS ###
    camus_dir = os.path.join(root_dir, 'CAMUS')
    train_dataset_list.append(SAMDataset(CAMUS(camus_dir, 'all'),
                                         image_transform=transforms)
                              )

    ## ct2usforkidneyseg ###
    usforkidney_dir = os.path.join(root_dir, 'ct2usforkidneyseg')
    train_dataset_list.append(SAMDataset(USForKidney(usforkidney_dir, 'all'),
                                         image_transform=transforms)
                              )

    ## BreastUS ###
    breastus_dir = os.path.join(root_dir, 'breast-ultrasound-images-dataset')
    train_dataset_list.append(SAMDataset(BreastUS(breastus_dir, 'all'),
                                         image_transform=transforms)
                              )

    ## USsimandsegm ###
    ussimandsegm_dir = os.path.join(root_dir, 'ussimandsegm')
    train_dataset_list.append(SAMDataset(USsimandsegmDataset(ussimandsegm_dir, 'all'),
                                         image_transform=transforms)
                              )

    ### USnerveseg ###
    usnerveseg_dir = os.path.join(root_dir, 'ultrasound-nerve-segmentation')
    train_dataset_list.append(SAMDataset(USnervesegDataset(usnerveseg_dir),
                                         image_transform=transforms)
                              )

    ### USThyroidDataset ###
    usthyroid_dir = os.path.join(root_dir, 'Thyroid Dataset')
    val_dataset_list.append(SAMDataset(USThyroidDataset(usthyroid_dir, 'all'),
                                       image_transform=transforms)
                            )

    ### FHPSAOPDataset ###
    fhpsaop_dir = os.path.join(root_dir, 'FH-PS-AOP')
    train_dataset_list.append(SAMDataset(FHPSAOPDataset(fhpsaop_dir, 'all'),
                                         image_transform=transforms)
                              )

    ### BUSI_WHU ###
    busi_whu_dir = os.path.join(root_dir, 'BUSI_WHU')
    train_dataset_list.append(SAMDataset(BUSI_WHU(busi_whu_dir, 'all'),
                                         image_transform=transforms)
                              )

    ### BUSCv ###
    buscv_dir = os.path.join(root_dir, 'BUSC-vckdnhtw26.1')
    train_dataset_list.append(SAMDataset(BUSCv(buscv_dir, 'all'),
                                         image_transform=transforms)
                              )

    ### FetalAbnominal ###
    fetalabnominal_dir = os.path.join(root_dir, 'Fetal_Abdominal')
    train_dataset_list.append(SAMDataset(FetalAbnominal(fetalabnominal_dir, 'all'),
                                         image_transform=transforms)
                              )

    ### BUS_UCLM ###
    busuclm_dir = os.path.join(root_dir, 'BUS-UCLM')
    train_dataset_list.append(SAMDataset(BUS_UCLM(busuclm_dir, 'all'),
                                         image_transform=transforms)
                              )

    ### RotatorCuffDataset ###
    rotatorcuff_dir = os.path.join(root_dir.replace('raw', 'processed'), 'CLEAN_RotatorCuff_DICOMs')
    train_dataset_list.append(SAMDataset(RotatorCuffDataset(rotatorcuff_dir, 'train'),
                                         image_transform=transforms)
                              )
    val_dataset_list.append(SAMDataset(RotatorCuffDataset(rotatorcuff_dir, 'test'),
                                       image_transform=transforms)
                            )

    return train_dataset_list, val_dataset_list


def load_colon_datasets(root_dir: str, transforms) -> tuple[list, list]:
    from datasets.endoscopy import BKAI_IGH_Dataset, CVCclinicdbDataset, KvasirSEG, PolypGen2021Dataset
    train_dataset_list = []
    val_dataset_list = []

    datasets_params = [
        (BKAI_IGH_Dataset, 'BKAI-IGH', {}),
        (CVCclinicdbDataset, 'cvcclinicdb', {}),
        (KvasirSEG, 'kvasirseg', {}),
        (PolypGen2021Dataset, 'PolypGen2021_MultiCenterData_v3', {})
    ]

    for dataset_cls, dataset_name, params in datasets_params:
        dataset_dir = os.path.join(root_dir, dataset_name)
        train_dataset_list.append(SAMDataset(dataset_cls(dataset_dir, 'train', **params),
                                             image_transform=transforms)
                                  )
        val_dataset_list.append(SAMDataset(dataset_cls(dataset_dir, 'test', **params),
                                           image_transform=transforms)
                                )

    return train_dataset_list, val_dataset_list


def main(args):
    MODEL_CFGS = {
        'small': ('sam2_hiera_s.yaml', 'sam2-checkpoints/sam2_hiera_small.pt'),
        'large': ('sam2_hiera_l.yaml', 'sam2-checkpoints/sam2_hiera_large.pt'),
        'small-adapter': ('sam2_hiera_s_adapted.yaml', 'sam2-checkpoints/sam2_hiera_small.pt'),
        'large-adapter': ('sam2_hiera_l_adapted.yaml', 'sam2-checkpoints/sam2_hiera_large.pt')
    }

    torch.set_float32_matmul_precision('high')
    root_dir = args.root_dir
    batch_size = args.batch_size

    model_cfg, checkpoint_path = MODEL_CFGS[args.model_arch]

    model = SAM2Model(checkpoint_path=checkpoint_path,
                      model_cfg=model_cfg,
                      learning_rate=args.learning_rate,
                      )
    model.freeze_all(freeze_mask_decoder=False,
                     freeze_adapter=False)
    train_dataset_list, val_dataset_list = load_colon_datasets(root_dir, model.predictor._transforms)
    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)

    _LOGGER.info(f"Train dataset size: {len(train_dataset)}")
    _LOGGER.info(f"Validation dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  persistent_workers=args.num_workers != 0,
                                  collate_fn=collate_dict_SAM)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=args.num_workers,
                                shuffle=False,
                                pin_memory=True,
                                persistent_workers=args.num_workers != 0,
                                collate_fn=collate_dict_SAM)

    fileout = ""
    llog_name = ''
    if args.project_name.strip() != '':
        fileout += args.project_name+'/'
        llog_name += args.project_name+'/'
    llog_name += model_cfg.split('.')[0]
    fileout += model_cfg.split('.')[0]
    fileout += f'/{datetime.datetime.now().strftime("%Y-%m-%d")}'
    fileout += '/sam2-{epoch:02d}-{val/iou:.2f}'
    checkpoint_callback = ModelCheckpoint(monitor='val/iou',
                                          filename=fileout,
                                          #   filename='testing',
                                          dirpath='checkpoints',
                                          auto_insert_metric_name=False,
                                          mode='max')
    tlogger = TensorBoardLogger('lightning_logs/',
                                name=llog_name)
    # log hyperparameters
    tlogger.log_hyperparams(args)

    trainer = Trainer(max_epochs=args.epochs,
                      num_sanity_val_steps=0,
                      enable_model_summary=False,
                      #   profiler='simple',
                      accelerator='gpu',
                      logger=tlogger,
                      precision="bf16-mixed",
                      callbacks=[checkpoint_callback, RichModelSummary()],
                      #   limit_train_batches=50,  # For debugging
                      #   limit_val_batches=50,
                      )
    trainer.validate(model, val_dataloader)
    if args.epochs > 0:
        trainer.fit(model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader
                    )


# Main script
if __name__ == "__main__":
    from rich.logging import RichHandler
    import argparse

    argparse = argparse.ArgumentParser()
    argparse.add_argument("--root_dir", type=str, default="data/raw")
    argparse.add_argument("--batch_size", type=int, default=8)
    argparse.add_argument("--epochs", type=int, default=30)
    argparse.add_argument("--num_workers", type=int, default=8)
    argparse.add_argument("--model_arch", type=str, default='small',
                          choices=['small', 'large', 'small-adapter', 'large-adapter'])
    argparse.add_argument("--learning_rate", type=float, default=2e-6)
    argparse.add_argument("--project-name", type=str, default='')

    args = argparse.parse_args()

    logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)],
                        format="%(message)s")
    logging.getLogger(__name__).setLevel(logging.INFO)

    main(args)
