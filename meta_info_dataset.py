import json
import sys

import csv
import collections
import SimpleITK as sitk

from class_modalities.datasets import DataManager
from class_modalities.utils import get_study_uid
from class_modalities.utils import roi2tmtv

config_name = 'config/default_config.json'

# read config file
with open(config_name) as f:
    config = json.load(f)

# path
csv_path = config['path']['csv_path']

thresholds = ['auto', 2.5, 'otsu']
dtypes = {'pet': sitk.sitkFloat32,
          'ct': sitk.sitkFloat32,
          'mask': sitk.sitkUInt8}

# Get Data
DM = DataManager(csv_path=csv_path)
dataset = collections.defaultdict(dict)
dataset['train'], dataset['val'], dataset['test'] = DM.get_train_val_test(wrap_with_dict=True)

header = ['STUDY_UID', 'subset']
for el in ['pet', 'ct', 'mask']:
    for info in ['_origin', '_size', '_spacing', '_direction']:
        header.append(el + info)
print(header)

print('images_meta_info.csv')
with open('/home/salim/Documents/DeepOncopole/data/meta_info/images_meta_info.csv', 'w') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(header)

    for subset, data in dataset.items():
        for i in range(len(data)):
            pet_id, ct_id, mask_id = data[i]['pet_img'], data[i]['ct_img'], data[i]['mask_img']
            study_id = get_study_uid(pet_id)
            print('[%4d  / %5d]: %s' % (i + 1, (i + 1) / len(data), study_id))
            # Total time: 0:00:04 (0.0828 s / it), eta : 0:25:13

            row = [study_id, subset]
            try:
                img = sitk.ReadImage(pet_id, dtypes['pet'])
                row = row + [img.GetOrigin(),
                             img.GetSize(),
                             img.GetSpacing(),
                             img.GetDirection()]
            except IOError:
                print('No file ' + pet_id)
                row = row + [None,
                             None,
                             None,
                             None]

            try:
                img = sitk.ReadImage(ct_id, dtypes['ct'])
                row = row + [img.GetOrigin(),
                             img.GetSize(),
                             img.GetSpacing(),
                             img.GetDirection()]
            except IOError:
                print('No file ' + ct_id)
                row = row + [None,
                             None,
                             None,
                             None]

            try:
                img = sitk.ReadImage(mask_id, dtypes['mask'])
                row = row + [img.GetOrigin(),
                             img.GetSize(),
                             img.GetSpacing(),
                             img.GetDirection()]
            except IOError:
                print('No file ' + mask_id)
                row = row + [None,
                             None,
                             None,
                             None]

            writer.writerow(row)

header = ['STUDY_UID', 'subset', 'size', 'spacing', 'n_roi', 'threshold', 'n_voxel_tumoral']
print(header)

header_roi = ['STUDY_UID', 'subset', 'size', 'spacing', 'num_roi', 'threshold', 'n_voxel_tumoral']
print(header_roi)

print('tmtv')
with open('/home/salim/Documents/DeepOncopole/data/meta_info/tmtv_meta_info.csv', 'w') as file:
    with open('/home/salim/Documents/DeepOncopole/data/meta_info/roi_mtv_meta_info.csv', 'w') as file_roi:

        writer = csv.writer(file, delimiter='\t')
        writer.writerow(header)

        writer_roi = csv.writer(file_roi, delimiter='\t')
        writer_roi.writerow(header_roi)

        for subset, data in dataset.items():
            for i in range(len(data)):
                pet_id, mask_id = data[i]['pet_img'], data[i]['mask_img']
                study_id = get_study_uid(pet_id)
                print(study_id)

                try:
                    pet_img = sitk.ReadImage(pet_id, dtypes['pet'])
                    mask_img = sitk.ReadImage(mask_id, dtypes['mask'])
                    spacing, size = mask_img.GetSpacing(), mask_img.GetSize()
                    if len(size) == 3:
                        n_roi = 1
                    else:
                        n_roi = size[-1]

                    rows = []
                    rows_roi = []
                    for t in thresholds:
                        tmtv, mtv_per_roi = roi2tmtv(mask_img, pet_img, threshold=t)

                        # ['STUDY_UID', 'subset', 'size', 'spacing', 'n_roi', 'threshold', 'n_voxel_tumoral']
                        rows.append([study_id, subset, size, spacing, n_roi, str(t), tmtv])

                        for num_roi, el in mtv_per_roi:
                            # ['STUDY_UID', 'subset', 'size', 'spacing', 'num_roi', 'threshold', 'n_voxel_tumoral']
                            rows_roi.append([study_id, subset, size, spacing, num_roi, str(t), el])

                    writer.writerows(rows)
                    writer_roi.writerows(rows_roi)
                except IOError:
                    print('No file ', pet_id, mask_id)
                except:
                    print(pet_id, mask_id)
                    print("Unexpected error:", sys.exc_info()[0])
