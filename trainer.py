import os
import warnings
from collections import OrderedDict

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from loss import YOLOloss
from predict import get_optimal_params, model_naylor_metrics


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: YOLOloss,
    train_data: DataLoader,
    cur_epoch: int,
    use_cuda: bool = True,
    scheduler=None,
    writer=None,
):
    model.train()
    summed_loss = 0
    bce_loss = 0
    mse_loss = 0
    gci_misclass_loss = 0
    zero_misclass_loss = 0
    f1 = 0
    recall = 0
    precision = 0
    batches = len(train_data)
    if use_cuda:
        if th.cuda.is_available():
            model.cuda()
        else:
            print("Warning: GPU not available, Running on CPU")
    if scheduler is not None:
        scheduler.step()
    for data, target in train_data:
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model(data)
        net_loss, loss_dict = loss.compute_loss(output, target)

        summed_loss += net_loss.item()
        bce_loss += loss_dict["BCE"]
        mse_loss += loss_dict["MSE"]
        gci_misclass_loss += loss_dict["GCI_Misclassification"]
        zero_misclass_loss += loss_dict["Zero_Misclassification"]
        f1 += loss_dict["f1_score"]
        recall += loss_dict["recall_score"]
        precision += loss_dict["precision_score"]

        net_loss.backward()
        # TODO: Gradient Clipping
        optimizer.step()
    metrics: OrderedDict = OrderedDict()
    metrics["net_loss"] = summed_loss / batches
    metrics["bce_loss"] = bce_loss / batches
    metrics["mse_loss"] = mse_loss / batches
    metrics["gci_misclassification_loss"] = gci_misclass_loss / batches
    metrics["zero_misclassification_loss"] = zero_misclass_loss / batches
    metrics["f1_score"] = f1 / batches
    metrics["recall_score"] = recall / batches
    metrics["precision_score"] = precision / batches

    return metrics


def test(
    model: nn.Module, loss, test_loader: DataLoader, use_cuda: bool = False, writer=None
):
    if use_cuda:
        model.cuda()
    model.eval()
    summed_loss = 0
    bce_loss = 0
    mse_loss = 0
    gci_misclass_loss = 0
    zero_misclass_loss = 0
    f1 = 0
    recall = 0
    precision = 0
    batches = len(test_loader)
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        with th.no_grad():
            output = model(data)

        net_loss, loss_dict = loss.compute_loss(output, target)

        summed_loss += net_loss.item()
        bce_loss += loss_dict["BCE"]
        mse_loss += loss_dict["MSE"]
        gci_misclass_loss += loss_dict["GCI_Misclassification"]
        zero_misclass_loss += loss_dict["Zero_Misclassification"]
        f1 += loss_dict["f1_score"]
        recall += loss_dict["recall_score"]
        precision += loss_dict["precision_score"]

    metrics: OrderedDict = OrderedDict()
    metrics["net_loss"] = summed_loss / batches
    metrics["bce_loss"] = bce_loss / batches
    metrics["mse_loss"] = mse_loss / batches
    metrics["gci_misclassification_loss"] = gci_misclass_loss / batches
    metrics["zero_misclassification_loss"] = zero_misclass_loss / batches
    metrics["f1_score"] = f1 / batches
    metrics["recall_score"] = recall / batches
    metrics["precision_score"] = precision / batches
    return metrics


def train_test_loop(config):
    config.setdefault("writer", None)
    train_data = config["train_data"]
    test_data = config["test_data"]
    model = config["model"]
    saver = config["saver"]
    use_cuda = config["use_cuda"]
    epochs = config["epochs"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    loss = config["loss"]
    writer = config["writer"]
    tee = config["tee"]
    test_interval = config["test_interval"]
    idr_interval = config["idr_interval"]
    model_dir = config["model_dir"]
    window = config["window"]
    stride = config["stride"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for epoch in range(1, epochs + 1):
            metrics = train(
                model,
                optimizer,
                loss,
                train_data,
                epoch,
                use_cuda,
                scheduler=scheduler,
                writer=writer,
            )
            tee.writeln(
                "Train: Epoch: {} ".format(epoch)
                + " ".join(("{}: {:.4f}".format(k, v) for k, v in metrics.items()))
            )

            writer.add_scalars(
                "loss/train",
                {
                    k: v
                    for k, v in metrics.items()
                    if k != "recall_score" or k != "precision_score" or k != "f1_score"
                },
                epoch,
            )
            writer.add_scalar("score/F/train", metrics["f1_score"], epoch)

            if epoch % test_interval == 0:
                metrics = test(model, loss, test_data, use_cuda, writer=writer)

                if epoch % idr_interval == 0:
                    validate_speechfolder = os.path.join(
                        model_dir, "validate", "speech"
                    )
                    validate_peaksfolder = os.path.join(model_dir, "validate", "peaks")
                    idr_params, _, _ = get_optimal_params(
                        model,
                        validate_speechfolder,
                        validate_peaksfolder,
                        window,
                        stride,
                        10,
                        use_cuda=False,
                    )

                    idr, mr, far, se = model_naylor_metrics(
                        model,
                        validate_speechfolder,
                        validate_peaksfolder,
                        window,
                        stride,
                        10,
                        use_cuda=False,
                        param_dict=idr_params,
                    )

                tee.writeln(
                    "\nVal:  Epoch: {} ".format(epoch)
                    + " ".join(("{}: {:.4f}".format(k, v) for k, v in metrics.items()))
                )

                if epoch % idr_interval == 0:
                    tee.writeln(
                        "Val:  Epoch: {} IDR: {} MR: {} FAR: {} SE: {}\nOptimal Params {}\n".format(
                            epoch, idr, mr, far, se, idr_params
                        )
                    )

                writer.add_scalars(
                    "loss/val",
                    {
                        k: v
                        for k, v in metrics.items()
                        if k != "recall_score"
                        or k != "precision_score"
                        or k != "f1_score"
                    },
                    epoch,
                )
                writer.add_scalar("score/F/val", metrics["f1_score"], epoch)

                checkpoint_params = {"epoch": epoch, **metrics}

                checkpoint = saver.create_checkpoint(
                    model.cpu(), optimizer, checkpoint_params
                )
                saver.save_checkpoint(
                    checkpoint, file_name="model_{}.pt".format(epoch), append_time=False
                )

