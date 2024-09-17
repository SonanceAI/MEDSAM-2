# pip install --upgrade dataset-tools

import dataset_tools as dtools

dst_dir = '../data/raw/endoscopy/lower/colonoscopy/'

# https://datasetninja.com/polypgen#download
# https://datasetninja.com/fine-grained-polyp
datasets_to_download = [
    # 'PolypGen',
    'Fine Grained Polyp',
    #  'CVC-ClinicDB'
]
for dataset_name in datasets_to_download:
    dtools.download(dataset=dataset_name, dst_dir=dst_dir)
    # TODO: delete the zip/tar files
