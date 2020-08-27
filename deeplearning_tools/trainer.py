import torch
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
import time
import os

from torch.utils.tensorboard import SummaryWriter


# loss_function = DiceLoss(sigmoid=True, squared_pred=True)
# optimizer = torch.optim.Adam(model.parameters(), 1e-4)


def MeanDice(model, data_loader, device):
    metric_sum = 0.0
    metric_count = 0
    for val_data in data_loader:
        val_inputs, val_labels = (
            val_data["image"].to(device),
            val_data["label"].to(device),
        )
        val_outputs = model(val_inputs)
        value = compute_meandice(val_outputs, val_inputs, sigmoid=True, logit_thresh=0.5)
        metric_count += len(value)
        metric_sum += value.sum().item()
    return metric_sum / metric_count


def fit(model, train_ds, val_ds, batch_size, epoch_num,
        loss_function, optimizer, device,
        callbacks=None, verbose=1):
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    writer = SummaryWriter()

    val_interval = 1  # do validation for every epoch,
    best_metric = float("-inf")
    best_metric_epoch = float("-inf")
    epoch_loss_values = list()
    metric_values = list()
    epoch_times = list()
    total_start = time.time()
    for epoch in range(epoch_num):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_ds) + step - 1)
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}"
                f" step time: {(time.time() - step_start):.4f}"
            )
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = model(val_inputs)
                    value = compute_meandice(val_outputs, val_inputs, sigmoid=True, logit_thresh=0.5)
                    metric_count += len(value)
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                writer.add_scalar('DiceMetric/val', metric, (epoch+1) * len(train_ds))
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}",
                    f" best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}",
                ),
        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
        epoch_times.append(time.time() - epoch_start)

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}",
        f" total time: {(time.time() - total_start):.4f}"
    ),
    return (
        epoch_num,
        time.time() - total_start,
        epoch_loss_values,
        metric_values,
        epoch_times,
    )



def train_process(train_ds, val_ds):
    # use batch_size=2 to load images and use RandCropByPosNegLabeld,
    # to generate 2 x 4 images for network training,
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=4, collate_fn=list_data_collate,
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, num_workers=4)
    device = torch.device("cuda:0")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    epoch_num = 600
    val_interval = 1  # do validation for every epoch,
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    epoch_times = list()
    total_start = time.time()
    for epoch in range(epoch_num):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}"
                f" step time: {(time.time() - step_start):.4f}"
            ),
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model
                    )
                    value = compute_meandice(
                        y_pred=val_outputs,
                        y=val_labels,
                        include_background=False,
                        to_onehot_y=True,
                        mutually_exclusive=True,
                    )
                    metric_count += len(value)
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}",
                    f" best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}",
                ),
        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
        epoch_times.append(time.time() - epoch_start)

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}",
        f" total time: {(time.time() - total_start):.4f}"
    ),
    return (
        epoch_num,
        time.time() - total_start,
        epoch_loss_values,
        metric_values,
        epoch_times,
    )