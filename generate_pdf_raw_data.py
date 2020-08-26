import sys
import json

import os
from class_modalities.utils import get_study_uid

import SimpleITK as sitk
from class_modalities.utils import mip
import numpy as np

import collections
from class_modalities.datasets import DataManager

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# read arg values
if len(sys.argv) == 2:
    config_name = 'config/default_config.json'
    MIP_folder = sys.argv[1]
elif len(sys.argv) == 3:
    config_name = sys.argv[1]
    MIP_folder = sys.argv[2]
else:
    config_name = 'config/default_config.json'
    MIP_folder = '/home/salim/Documents/DeepOncopole/data/MIP_dataset'

# read config file
with open(config_name) as f:
    config = json.load(f)

# path
csv_path = config['path']['csv_path']  # /media/salim/DD 2To/AHL2011_NIFTI/AHL2011_PET0_NIFTI.csv'


# Get Data
DM = DataManager(csv_path=csv_path)
dataset = collections.defaultdict(dict)
dataset['train'], dataset['val'], dataset['test'] = DM.get_train_val_test(wrap_with_dict=True)

for subset_type, data in dataset.items():
    print(subset_type)
    # path to pdf to generate
    filename = os.path.join(MIP_folder, subset_type, 'MIP_raw_{}_data.pdf'.format(subset_type))
    print('folder :', os.path.join(MIP_folder, subset_type))
    print('filename :', filename)
    if not os.path.exists(os.path.join(MIP_folder, subset_type)):
        os.makedirs(os.path.join(MIP_folder, subset_type))
        print(os.path.join(MIP_folder, subset_type), 'folder created')

    with PdfPages(filename) as pdf:

        for step, img_path in dataset:
            study_id = get_study_uid(img_path['pet_img'])
            print(step, study_id)

            pet_img = sitk.ReadImage(img_path['pet_img'], sitk.sitkFloat32)

            mip1, mip2 = mip(pet_img, threshold=2.5)
            img_plot = np.hstack((mip1, mip2))

            plt.imshow(img_plot, plt.cm.plasma)
            # plt.axis('off')
            plt.colorbar()
            plt.title('PET', fontsize=20)
            plt.show()

            # Plot
            f = plt.figure(figsize=(15, 10))
            f.suptitle(study_id, fontsize=15)

            plt.subplot(121)
            plt.axis('off')
            plt.title('PET/CT', fontsize=20)

            plt.subplot(122)
            plt.axis('off')
            plt.title('PET/CT + Segmentation', fontsize=20)

            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
