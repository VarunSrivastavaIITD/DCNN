import json
import math
import os
import random
import sys

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

# from loader import create_train_validate_test_data
from loss import YOLOloss
from model_summarize import torch_summarize_df
from saver import Saver
from trainer import test, train_test_loop
from utils import Tee, get_model_dir, copy_files
from dataloader import create_train_validate_test_dataloader

# INFO: Set random seeds
np.random.seed(42)
th.manual_seed(42)
th.cuda.manual_seed_all(42)
random.seed(42)


class SELUNet(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()

        c1 = nn.Sequential(nn.Conv1d(1, 10, 5, padding=4), nn.MaxPool1d(2), nn.SELU())

        c2 = nn.Sequential(
            nn.Conv1d(10, 15, 3, padding=2, dilation=2), nn.MaxPool1d(2), nn.SELU()
        )

        c3 = nn.Sequential(
            nn.Conv1d(15, 15, 3, padding=4, dilation=4), nn.MaxPool1d(2), nn.SELU()
        )

        self.feature_extractor = nn.Sequential(c1, c2, c3)

        self.classifier = nn.Sequential(
            nn.Linear(150, 50),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(50, 10),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(10, 1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(150, 50),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(50, 10),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(10, 1),
            nn.Hardtanh(0, 80 - 1),
        )

        self._initialize_submodules()

    def _initialize_submodules(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.kaiming_normal(m.weight.data)
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(1.0 / n))
            elif isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.out_channels
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1.0 / n))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x1 = self.regressor(x)
        x2 = self.classifier(x)
        y = th.squeeze(th.cat((x1, x2), 1))
        return y


def config():
    def hyperparams_config():
        hpdict = {}
        hpdict["epochs"] = 40
        hpdict["stride"] = 1
        hpdict["batch_size"] = 2
        hpdict["window"] = 80
        hpdict["input_channels"] = 1
        hpdict["lr"] = 2e-3
        hpdict["lr_decay"] = 0.9
        hpdict["lr_decay_epochs"] = 10
        hpdict["bce_weight"] = 1
        hpdict["mse_weight"] = 0.1
        hpdict["gci_misclassification_weight"] = 0.1
        hpdict["zero_misclassification_weight"] = 0.1
        hpdict["threshold"] = 0.5
        hpdict["alphadropout"] = 0

        return hpdict

    def train_config():
        ddict = {}
        ddict["train_ratio"] = 0.3
        ddict["validate_ratio"] = 0.1
        ddict["test_ratio"] = 0.6
        ddict["idr_interval"] = 40
        ddict["test_interval"] = 5
        ddict["model_dir"] = get_model_dir(model_prefix="model")
        # ddict["speech_dir"] = "speech"
        # ddict["peaks_dir"] = "peaks"
        ddict["speech_npz_file"] = "example_speech.npz"
        ddict["egg_npz_file"] = "example_egg.npz"
        ddict["window"] = 80
        ddict["use_cuda"] = True
        ddict["load_test"] = False
        ddict["model_resume_file"] = None
        ddict["writer_dir"] = os.path.join(ddict["model_dir"], "runs")
        ddict["tee_file"] = os.path.join(ddict["model_dir"], "log.txt")
        ddict["save_dir"] = os.path.join(ddict["model_dir"], "saved_models")
        ddict["hyperparams_config_file"] = os.path.join(
            ddict["model_dir"], "hyperparams.txt"
        )
        ddict["train_config_file"] = os.path.join(
            ddict["model_dir"], "train_config.txt"
        )
        ddict["train_config_file"] = os.path.join(
            ddict["model_dir"], "train_config.txt"
        )
        ddict["model_summary_file"] = os.path.join(
            ddict["model_dir"], "model_summary.txt"
        )

        return ddict

    hpdict, ddict = hyperparams_config(), train_config()
    return {**hpdict, **ddict}, hpdict, ddict


def dump_config(model, netconfig, hyperparams_config, data_config):
    os.makedirs(netconfig["model_dir"], exist_ok=True)
    model_summary = torch_summarize_df(
        (netconfig["input_channels"], netconfig["window"]), model
    )
    with open(netconfig["train_config_file"], "w") as f:
        json.dump(data_config, f)
    with open(netconfig["hyperparams_config_file"], "w") as f:
        json.dump(hyperparams_config, f)
    with open(netconfig["model_summary_file"], "w") as f:
        f.write(str(model_summary))


def construct_data(netconfig):
    # return create_train_validate_test_data(
    #     netconfig["speech_dir"],
    #     netconfig["peaks_dir"],
    #     netconfig["model_dir"],
    #     split={
    #         "train": netconfig["train_ratio"],
    #         "validate": netconfig["validate_ratio"],
    #         "test": netconfig["test_ratio"],
    #     },
    #     batch_size=netconfig["batch_size"],
    #     stride=netconfig["stride"],
    #     symlink=True,
    #     load_test=netconfig["load_test"],
    # )
    pass


def construct_loss(netconfig):
    return YOLOloss(
        netconfig["bce_weight"],
        netconfig["mse_weight"],
        netconfig["gci_misclassification_weight"],
        netconfig["zero_misclassification_weight"],
        netconfig["threshold"],
    )


def load_model(model, modelfile, map_location="cuda"):
    if modelfile is not None:
        checkpoint: dict = th.load(modelfile, map_location=map_location)
        model.load_state_dict(checkpoint["model_dict"])
    return model


def main():

    netconfig, hyperparams_config, data_config = config()
    model = SELUNet(dropout=netconfig["alphadropout"])
    model = load_model(model, netconfig["model_resume_file"])

    dump_config(model, netconfig, hyperparams_config, data_config)
    # copy_files(netconfig["model_dir"])

    train_data, validate_data, test_data = create_train_validate_test_dataloader(
        netconfig
    )

    saver = Saver(netconfig["save_dir"])
    use_cuda = netconfig["use_cuda"]
    writer = SummaryWriter(netconfig["writer_dir"])
    tee = Tee(netconfig["tee_file"])

    optimizer = optim.Adamax(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, netconfig["lr_decay_epochs"], netconfig["lr_decay"]
    )
    loss = construct_loss(netconfig)

    finalconfig = {
        "train_data": train_data,
        "test_data": validate_data,
        "model": model,
        "saver": saver,
        "use_cuda": use_cuda,
        "epochs": netconfig["epochs"],
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss": loss,
        "writer": writer,
        "tee": tee,
        "model_dir": netconfig["model_dir"],
        "test_interval": netconfig["test_interval"],
        "window": netconfig["window"],
        "stride": netconfig["stride"],
        "idr_interval": netconfig["idr_interval"],
    }

    train_test_loop(finalconfig)

    if test_data:
        metrics = test(model, loss, test_data, use_cuda)

        tee.writeln(
            "Test: "
            + " ".join(("{: >5}: {:.4f}".format(k, v) for k, v in metrics.items()))
        )


if __name__ == "__main__":
    main()
