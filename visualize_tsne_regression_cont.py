import argparse
import os
import random
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
from sklearn.manifold import TSNE

from dataloader import AudioDataset
from main import SELUNet, config, load_model
from saver import Saver
from datetime import datetime as dt

plt.switch_backend("qt5agg")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
SMALL_SIZE = 14
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# INFO: Set random seeds
np.random.seed(42)
th.manual_seed(42)
th.cuda.manual_seed_all(42)
random.seed(42)


def main():

    netconfig, _, _ = config()
    model = SELUNet(dropout=netconfig["alphadropout"])
    netconfig["model_resume_file"] = "model3/saved_models/model_20.pt"
    model = load_model(model, netconfig["model_resume_file"])
    model.eval()

    train_data = AudioDataset(
        "bdl_speech.npz", "bdl_egg.npz", netconfig["window"], netconfig["window"]
    )
    data = [train_data[i] for i in range(10)]
    speech = th.cat([d[0] for d in data])
    peak_arr = th.cat([d[1] for d in data])

    idx = np.random.choice(len(speech), 2000, replace=False)
    speech = speech[idx].unsqueeze_(1)

    with th.no_grad():
        features = model.feature_extractor(speech)

    print(f"speech shape {speech.shape} features shape {features.shape}")
    features = features.view(features.size(0), -1).numpy()

    labels = peak_arr[:, 1][idx].numpy()

    features_tsne = TSNE(perplexity=70, n_iter=2000).fit_transform(features)
    print(f"features tsne shape {features_tsne.shape}")

    positive_idx = labels == 1
    negative_idx = labels == 0

    print(
        f"positive samples: {np.count_nonzero(positive_idx)} negative samples: {np.count_nonzero(negative_idx)}"
    )

    positive_coords = features_tsne[positive_idx]
    negative_coords = features_tsne[negative_idx]
    positive_distances = peak_arr[:, 0][idx].numpy()[positive_idx]

    # bins = np.linspace(0, netconfig["window"], 5)
    # markers = [".", "v", "^", "*"]
    # colors = ["b", "r", "k", "m"]
    # inds = np.digitize(positive_distances, bins)

    fig, ax = plt.subplots()

    # ax.set_title("TSNE L2 plot")
    sc = ax.scatter(
        positive_coords[:, 0],
        positive_coords[:, 1],
        c=positive_distances / (netconfig["window"] - 1),
        # label=f"++",
        marker="o",
        cmap="viridis",
    )
    # ax.scatter(
    #     negative_coords[:, 0], negative_coords[:, 1], c="g", label="--", marker="."
    # )
    cbar = plt.colorbar(sc)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["$1$", "$w_d$"])

    # ax.legend(
    #     ncol=2,
    #     framealpha=0.2,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.15),
    #     mode=None,
    #     markerscale=4,
    # )
    ax.tick_params(
        axis="both",
        which="both",
        bottom="off",
        top="off",
        labelbottom="off",
        right="off",
        left="off",
        labelleft="off",
    )
    fig.set_tight_layout(True)
    # ax.set_xlabel("Dimension 1")
    # ax.set_ylabel("Dimension 2")

    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    # plt.show()
    model_folder = os.path.join(netconfig["model_resume_file"].split("/")[0], "plots")
    os.makedirs(model_folder, exist_ok=True)

    curtime = dt.now()
    plt.savefig(
        os.path.join(
            model_folder,
            f"fig_{curtime.day}_{curtime.hour}_{curtime.minute}_{curtime.second}.eps",
        ),
        format="eps",
        dpi=300,
    )


if __name__ == "__main__":
    main()
