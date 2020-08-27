import sys
import argparse
import json

import os
from datetime import datetime

import glob
import shutil
import tempfile

import numpy as np
from class_modalities.datasets import DataManager
from class_modalities.transforms import LoadNifti, Roi2Mask, ResampleReshapeAlign, Sitk2Numpy, ConcatModality

import pytorch_lightning
import torch

import monai
from monai.config import print_config
from monai.data import CacheDataset, list_data_collate
from monai.losses import DiceLoss

from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadNiftid,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    RandAffined,
    ToTensord,
    Activationsd,
    AsDiscreted,
)

from ignite.metrics import Accuracy, Precision, Recall
from monai.metrics import compute_meandice

from monai.inferers import SimpleInferer
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
)
from monai.utils import set_determinism

print_config()


def main(config):
    now = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    # path
    csv_path = config['path']['csv_path']

    trained_model_path = config['path']['trained_model_path']  # if None, trained from scratch
    training_model_folder = os.path.join(config['path']['training_model_folder'], now)  # '/path/to/folder'
    if not os.path.exists(training_model_folder):
        os.makedirs(training_model_folder)
    logdir = os.path.join(training_model_folder, 'logs')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # PET CT scan params
    image_shape = tuple(config['preprocessing']['image_shape'])  # (x, y, z)
    in_channels = config['preprocessing']['in_channels']
    voxel_spacing = tuple(config['preprocessing']['voxel_spacing'])  # (4.8, 4.8, 4.8)  # in millimeter, (x, y, z)
    data_augment = config['preprocessing']['data_augment']  # True  # for training dataset only
    resize = config['preprocessing']['resize']  # True  # not use yet
    origin = config['preprocessing']['origin']  # how to set the new origin
    normalize = config['preprocessing']['normalize']  # True  # whether or not to normalize the inputs
    number_class = config['preprocessing']['number_class']  # 2

    # CNN params
    architecture = config['model']['architecture']  # 'unet' or 'vnet'

    cnn_params = config['model'][architecture]['cnn_params']
    # transform list to tuple
    for key, value in cnn_params.items():
        if isinstance(value, list):
            cnn_params[key] = tuple(value)

    # Training params
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    shuffle = config['training']['shuffle']
    opt_params = config['training']["optimizer"]["opt_params"]

    # Get Data
    DM = DataManager(csv_path=csv_path)
    train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)

    # Input preprocessing
    # use data augmentation for training
    train_transforms = Compose(
        [  # read img + meta info
            LoadNifti(keys=["pet_img", "ct_img", "mask_img"]),
            Roi2Mask(keys=['pet_img', 'mask_img'], method='otsu', tval=0.0, idx_channel=0),
            ResampleReshapeAlign(target_shape=image_shape, target_voxel_spacing=voxel_spacing,
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
            # normalize input
            ScaleIntensityRanged(
                keys=["pet_img"], a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True,
            ),
            ScaleIntensityRanged(
                keys=["ct_img"], a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True,
            ),
            # Prepare for neural network
            ConcatModality(keys=['pet_img', 'ct_img']),
            AddChanneld(keys=["mask_img"]),  # Add channel to the first axis
            ToTensord(keys=["image", "mask_img"]),
        ]
    )
    # without data augmentation for validation
    val_transforms = Compose(
        [  # read img + meta info
            LoadNifti(keys=["pet_img", "ct_img", "mask_img"]),
            Roi2Mask(keys=['pet_img', 'mask_img'], method='otsu', tval=0.0, idx_channel=0),
            ResampleReshapeAlign(target_shape=image_shape, target_voxel_spacing=voxel_spacing,
                                 keys=['pet_img', "ct_img", 'mask_img'],
                                 origin='head', origin_key='pet_img'),
            Sitk2Numpy(keys=['pet_img', 'ct_img', 'mask_img']),
            # normalize input
            ScaleIntensityRanged(
                keys=["pet_img"], a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True,
            ),
            ScaleIntensityRanged(
                keys=["ct_img"], a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True,
            ),
            # Prepare for neural network
            ConcatModality(keys=['pet_img', 'ct_img']),
            AddChanneld(keys=["mask_img"]),  # Add channel to the first axis
            ToTensord(keys=["image", "mask_img"]),
        ]
    )

    # create a training data loader
    train_ds = monai.data.CacheDataset(data=train_images_paths, transform=train_transforms, cache_rate=0.5)
    # use batch_size=2 to load images to generate 2 x 4 images for network training
    train_loader = monai.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    # create a validation data loader
    val_ds = monai.data.CacheDataset(data=val_images_paths, transform=val_transforms, cache_rate=1.0)
    val_loader = monai.data.DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    # Model
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(
        dimensions=3,  # 3D
        in_channels=in_channels,
        out_channels=1,
        kernel_size=5,
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)

    # training
    val_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True),
        ]
    )
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir="./runs/", output_transform=lambda x: None),
        # TensorBoardImageHandler(
        #     log_dir="./runs/",
        #     batch_transform=lambda x: (x["image"], x["label"]),
        #     output_transform=lambda x: x["pred"],
        # ),
        CheckpointSaver(save_dir="./runs/", save_dict={"net": net, "opt": opt}, save_key_metric=True),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SimpleInferer(),
        post_transform=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=True, output_transform=lambda x: (x["pred"], x["label"]))
        },
        additional_metrics={"val_precision": Precision(output_transform=lambda x: (x["pred"], x["label"])),
                            "val_recall": Recall(output_transform=lambda x: (x["pred"], x["label"]))},
        val_handlers=val_handlers,
        # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP evaluation
        # amp=True if monai.config.get_torch_version_tuple() >= (1, 6) else False,
    )

    train_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True),
        ]
    )
    train_handlers = [
        # LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
        TensorBoardStatsHandler(log_dir="./runs/", tag_name="train_loss", output_transform=lambda x: x["loss"]),
        CheckpointSaver(save_dir="./runs/", save_dict={"net": net, "opt": opt}, save_interval=2, epoch_level=True),
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=5,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss,
        inferer=SimpleInferer(),
        post_transform=train_post_transforms,
        key_train_metric={"train_acc": Accuracy(output_transform=lambda x: (x["pred"], x["label"]))},
        additional_metrics={"train_precision": Precision(output_transform=lambda x: (x["pred"], x["label"])),
                            "train_recall": Recall(output_transform=lambda x: (x["pred"], x["label"]))},
        train_handlers=train_handlers,
        # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP training
        amp=True if monai.config.get_torch_version_tuple() >= (1, 6) else False,
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/default_config.json', type=str,
                        help="json config file")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    main(config)



