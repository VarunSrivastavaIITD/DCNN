import numpy as np
import sys
from scipy.ndimage import label, find_objects
import matplotlib.pyplot as plt
from metrics import corrected_naylor_metrics, adjusted_naylor_metrics


def cluster(gcilocs,
            weights,
            threshold=0.9,
            samples_per_bin=4,
            histogram_count_threshold=20):
    if np.max(weights) > 1 and threshold < 1:
        raise ValueError('Weights are unnormalized and threshold < 1')

    if len(weights) != len(gcilocs):
        raise ValueError('Unequal array sizes')

    sind = np.argsort(gcilocs)
    gcilocs = gcilocs[sind]
    weights = weights[sind]

    tind = weights > threshold
    gcilocs = gcilocs[tind]
    weights = weights[tind]

    numsamples = len(gcilocs)
    numbins = numsamples // samples_per_bin

    hist, bin_edges = np.histogram(gcilocs, bins=numbins, weights=weights)
    # hist, bin_edges = np.histogram(gcilocs, bins=numbins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    cntid = hist < histogram_count_threshold
    hist[cntid] = 0

    regions = find_objects(label(hist)[0])

    GCI = [np.average(bin_centers[r], weights=hist[r]) for r in regions]
    # GCI = [np.average(bin_centers[r]) for r in regions]

    return GCI


def main():
    gt = np.load('ModelPredictions/ground_truth.npy')
    pt = np.load('ModelPredictions/predicted.npy')

    gx = gt[0, :]
    gy = gt[1, :]

    px = pt[0, :]
    py = pt[1, :]

    fs = 16000

    gci = np.array(
        cluster(
            px,
            py,
            threshold=0.7,
            samples_per_bin=25,
            histogram_count_threshold=20))
    predicted_gci_time = gci / fs
    target_gci_time = np.nonzero(gy)[0] / fs

    gci = np.round(gci).astype(np.int64)
    gcilocs = np.zeros_like(gx)
    gcilocs[gci] = 1

    print(corrected_naylor_metrics(target_gci_time, predicted_gci_time))
    print(adjusted_naylor_metrics(target_gci_time, predicted_gci_time))

    # plt.plot(gx, gy, color='b')
    # plt.plot(gx, gcilocs, color='r')
    # plt.show()


if __name__ == "__main__":
    main()
