import torch
import monai

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
    for data in data_loader:
        inputs, labels = (
            data["image"].to(device),
            data["label"].to(device),
        )
        outputs = model(inputs)
        value = compute_meandice(outputs, inputs, sigmoid=True, logit_thresh=0.5)
        metric_count += len(value)
        metric_sum += value.sum().item()
    return metric_sum / metric_count


def fit(model, train_ds, val_ds, batch_size, epoch_num,
        loss_function, optimizer, device, root_dir,
        callbacks=None, verbose=1):
    # train_loader = torch.utils.data.DataLoader(
    #     train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    # )
    # val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    train_loader = monai.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = monai.data.DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    # tensorboard
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
                f"Epoch: [{epoch + 1}], [{step}/{len(train_ds) // train_loader.batch_size}], train_loss: {loss.item():.4f} step time: {(time.time() - step_start):.4f} "
            )  #  ETA: 0:01:18
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
                writer.add_scalar('DiceMetric/val', metric, (epoch + 1) * len(train_ds))
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"),
                    )
                    # torch.save({
                    #             'epoch': EPOCH,
                    #             'model_state_dict': net.state_dict(),
                    #             'optimizer_state_dict': optimizer.state_dict(),
                    #             'loss': LOSS,
                    #             }, PATH)
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
