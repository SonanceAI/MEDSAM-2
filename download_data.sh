#!/bin/bash

# https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md

OUTPATH=data/raw

KAGGLE_DATASETS=("aryashah2k/breast-ultrasound-images-dataset" "ignaciorlando/ussimandsegm" "siatsyx/ct2usforkidneyseg")
for dataset in ${KAGGLE_DATASETS[@]};
do
    foldername=$(echo $dataset | cut -d'/' -f2)
    kaggle datasets download -d $dataset -p $OUTPATH/$foldername --unzip
done

kaggle competitions download -c ultrasound-nerve-segmentation -p /tmp/
unzip -d $OUTPATH/ultrasound-nerve-segmentation /tmp/ultrasound-nerve-segmentation.zip && rm /tmp/ultrasound-nerve-segmentation.zip

wget -O /tmp/CAMUS.zip https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/6373703d73e9f0047faa1bc8/download
unzip -d $OUTPATH/CAMUS /tmp/CAMUS.zip && rm /tmp/CAMUS.zip

wget -O /tmp/FH-PS-AOP.zip https://zenodo.org/api/records/7851339/files-archive
unzip -d /tmp/FH-PS-AOP /tmp/FH-PS-AOP.zip && rm /tmp/FH-PS-AOP.zip
unzip -d $OUTPATH/FH-PS-AOP "/tmp/FH-PS-AOP/Pubic Symphysis-Fetal Head Segmentation and Angle of Progression.zip" && rm -r /tmp/FH-PS-AOP

wget -O /tmp/HC.zip https://zenodo.org/api/records/1327317/files-archive
unzip -d $OUTPATH/HC "/tmp/HC.zip" && rm /tmp/HC.zip

# BUSC Dataset (https://data.mendeley.com/datasets/vckdnhtw26/1)
wget -O /tmp/vckdnhtw26-1.zip https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/vckdnhtw26-1.zip
unzip -d $OUTPATH/BUSC-vckdnhtw26.1 /tmp/vckdnhtw26-1.zip && rm /tmp/vckdnhtw26-1.zip && mv "$OUTPATH/BUSC-vckdnhtw26.1/BUSC Dataset/BUSC_Dataset" "$OUTPATH/BUSC-vckdnhtw26.1/." && rmdir "$OUTPATH/BUSC-vckdnhtw26.1/BUSC Dataset"

# BUS-UCLM (https://data.mendeley.com/datasets/7fvgj4jsp7/1)
wget -O /tmp/7fvgj4jsp7-1.zip https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/7fvgj4jsp7-1.zip
unzip -d $OUTPATH/BUS-UCLM /tmp/7fvgj4jsp7-1.zip && rm /tmp/7fvgj4jsp7-1.zip

# BUS_UC (https://data.mendeley.com/datasets/3ksd7w7jkx/1)
wget -O /tmp/BUS_UC.zip https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/3ksd7w7jkx-1.zip
unzip -d $OUTPATH/BUS_UC /tmp/BUS_UC.zip && rm /tmp/BUS_UC.zip

# Fetal_Abdominal https://data.mendeley.com/datasets/4gcpm9dsc3/1
wget -O /tmp/Fetal_Abdominal.zip https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/4gcpm9dsc3-1.zip
unzip -d $OUTPATH/Fetal_Abdominal /tmp/Fetal_Abdominal.zip && rm /tmp/Fetal_Abdominal.zip # TODO: extract the zip inside the zip