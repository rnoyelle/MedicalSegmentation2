
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadNiftid,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    RandAffined,
    ToTensord,
)
from monai.utils import set_determinism
from .transforms import *


class LymphomaDetectionDataset(object):

    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_dict = self.images[idx]
        if self.transforms is not None :
            return self.transforms(img_dict)
        else:
            return img_dict

    def default_transform(self, train=False):
        target_shape = (128, 128, 256)  # (x, y, z)
        target_voxel_spacing = (4.8, 4.8, 4.8)  # (x, y, z)
        if train :
            return Compose(
                [  # read img + meta info
                    LoadNifti(keys=["pet_img", "ct_img", "mask_img"]),
                    Roi2Mask(keys=['pet_img', 'mask_img'], method='otsu', tval=0.0, idx_channel=0),
                    ResampleReshapeAlign(target_shape, target_voxel_spacing,
                                         keys=['pet_img', "ct_img", 'mask_img'],
                                         origin='head', origin_key='pet_img'),
                    Sitk2Numpy(keys=['pet_img', 'ct_img', 'mask_img']),
                    # user can also add other random transforms
                    RandAffined(keys=("pet_img", "ct_img", "mask_img"),
                                spatial_size=None, prob=0.4,
                                rotate_range=(0, np.pi / 30, np.pi / 15), shear_range=None,
                                translate_range=(10, 10, 10), scale_range=(0.1, 0.1, 0.1),
                                mode=("bilinear", "bilinear", "nearest"),
                                padding_mode="border"),
                    # semantic to instance segmentation
                    ConnectedComponent(keys='mask_img', channels_first=True, exclude_background=True),
                    # Add bouding Box
                    GenerateBbox(keys='mask_img'),
                    # normalize input
                    ScaleIntensityRanged(
                        keys=["pet_img"], a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True,
                    ),
                    ScaleIntensityRanged(
                        keys=["ct_img"], a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True,
                    ),
                    # Prepare for neural network
                    ConcatModality(keys=['pet_img', 'ct_img']),
                    ToTensord(keys=["image", "mask_img"]),
                ]
            )
        else:
            return Compose(
                [  # read img + meta info
                    LoadNifti(keys=["pet_img", "ct_img", "mask_img"]),
                    Roi2Mask(keys=['pet_img', 'mask_img'], method='otsu', tval=0.0, idx_channel=0),
                    ResampleReshapeAlign(target_shape, target_voxel_spacing,
                                         keys=['pet_img', "ct_img", 'mask_img'],
                                         origin='head', origin_key='pet_img'),
                    Sitk2Numpy(keys=['pet_img', 'ct_img', 'mask_img']),
                    # semantic to instance segmentation
                    ConnectedComponent(keys='mask_img', channels_first=True, exclude_background=True),
                    # Add bouding Box
                    GenerateBbox(keys='mask_img'),
                    # normalize input
                    ScaleIntensityRanged(
                        keys=["pet_img"], a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True,
                    ),
                    ScaleIntensityRanged(
                        keys=["ct_img"], a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True,
                    ),
                    # Prepare for neural network
                    ConcatModality(keys=['pet_img', 'ct_img']),
                    ToTensord(keys=["image", "mask_img"]),
                ]
            )

