#!/bin/bash

# Paper with a list of colonoscopy datasets: https://www.nature.com/articles/s41597-024-03359-0

# https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md

OUTPATH=../data/raw/endoscopy
OUTPATH_COL=$OUTPATH/lower/colonoscopy
mkdir -p $OUTPATH
mkdir -p $OUTPATH_COL

KAGGLE_DATASETS=("newslab/cholecseg8k" "balraj98/cvcclinicdb" "debeshjha1/kvasirseg")
for dataset in "${KAGGLE_DATASETS[@]}";
do
    foldername=$(echo $dataset | cut -d'/' -f2)
    kaggle datasets download -d $dataset -p $OUTPATH_COL/$foldername --unzip
done

kaggle competitions download -c bkai-igh-neopolyp -p /tmp/
unzip -d $OUTPATH_COL/BKAI-IGH /tmp/bkai-igh-neopolyp.zip && rm /tmp/bkai-igh-neopolyp.zip

# array of space-separated strings where the first element is the output name in the second is the URL
# HTTP_DATASETS_COL=(
#     # "Kvasir https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip" # Kvasir # Multi-class problem
#     Kvasir-seg https://datasets.simula.no/downloads/kvasir-seg.zip # Kvasir-seg # Segmentation # https://datasets.simula.no/kvasir-seg/
#                   )
# for dataset in "${HTTP_DATASETS_COL[@]}";
# do
#     foldername=$(echo $dataset | cut -d' ' -f1)
#     url=$(echo $dataset | cut -d' ' -f2)
#     wget -O /tmp/$foldername.zip $url
#     unzip -d $OUTPATH_COL/$foldername /tmp/$foldername.zip && rm /tmp/$foldername.zip
# done

### Data that needs login ###
# https://www.synapse.org/Synapse:syn45200214 # PolypGen2021_MultiCenterData_v3
# https://github.com/dashishi/LDPolypVideo-Benchmark?tab=readme-ov-file


# https://www.nature.com/articles/s41597-024-03359-0#ref-CR21: This dataset has 2.7M video frames, but only has bounding boxes. 
# We might want to use it somehow for segmentation, but it will need some work to get the segmentation masks or some adaptation.