import argparse
import os
import random
import warnings
from glob import glob

import numpy as np
import torch as th
import torch.nn.functional as F
from utils import strided_app
from torch_utils import to_variable
import matplotlib.pyplot as plt

from saver import Saver
from main import SELUWeightNet
# INFO: Set random seeds
np.random.seed(42)
th.manual_seed(42)
th.cuda.manual_seed_all(42)
random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--speechfolder',
        type=str,
        default='test/speech',
        help='data directory containing speech files')

    parser.add_argument(
        '--peaksfolder',
        type=str,
        default='test/peaks',
        help='data directory containing peak files')
    parser.add_argument(
        '--window',
        type=int,
        default=80,
        help='window size for the overlapping sub arrays')
    parser.add_argument(
        '--stride', type=int, default=1, help='stride of the moving window')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='model_15.pt',
        help='checkpoint file containing the model to use for prediction')
    parser.add_argument(
        '--model_dir',
        default='MyFinalModel',
        type=str,
        help='Directory containing checkpoint files')
    parser.add_argument(
        '--use_cuda', type=bool, default=False, help='use gpu for inference')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='threshold for discerning peaks')
    parser.add_argument(
        '--prediction_dir',
        type=str,
        default='ModelPredictions',
        help='Base Directory to store model predictions')
    args = parser.parse_args()
    return args


def create_dataset(speechfolder, peaksfolder, window, stride, numfiles=10):
    speechfiles = sorted(glob(os.path.join(speechfolder, '*.npy')))[-numfiles:]
    peakfiles = sorted(glob(os.path.join(peaksfolder, '*.npy')))[-numfiles:]

    speech_data = [np.load(f) for f in speechfiles]
    peak_data = [np.load(f) for f in peakfiles]

    speech_data = np.concatenate(speech_data)
    peak_data = np.concatenate(peak_data)
    indices = np.arange(len(speech_data))

    speech_windowed_data = strided_app(speech_data, window, stride)
    peak_windowed_data = strided_app(peak_data, window, stride)
    indices = strided_app(indices, window, stride)

    peak_distance = np.array([
        np.nonzero(t)[0][0] if len(np.nonzero(t)[0]) != 0 else -1
        for t in peak_windowed_data
    ])

    peak_indicator = (peak_distance != -1) * 1.0

    return speech_windowed_data, peak_distance, peak_indicator, indices, peak_data


def main():
    args = parse_args()
    speech_windowed_data, peak_distance, peak_indicator, indices, actual_gci_locations = create_dataset(
        args.speechfolder, args.peaksfolder, args.window, args.stride, 10)
    saver = Saver(args.model_dir)
    model = SELUWeightNet
    model, _, params_dict = saver.load_checkpoint(
        model, file_name=args.model_name)
    model.eval()

    input = to_variable(
        th.from_numpy(
            np.expand_dims(speech_windowed_data, 1).astype(np.float32)),
        args.use_cuda, True)

    with warnings.catch_warnings():
        if args.use_cuda:
            model = model.cuda()
        warnings.simplefilter('ignore')
        prediction = model(input)

    predicted_peak_indicator = F.sigmoid(prediction[:, 1]).data.numpy()
    predicted_peak_distance = (prediction[:, 0]).data.numpy().astype(np.int32)

    predicted_peak_indicator_indices = predicted_peak_indicator > args.threshold

    predicted_peak_indicator = predicted_peak_indicator[
        predicted_peak_indicator_indices].ravel()
    predicted_peak_distance = predicted_peak_distance[
        predicted_peak_indicator_indices].ravel()
    indices = indices[predicted_peak_indicator_indices]

    assert (len(indices) == len(predicted_peak_distance))
    assert (len(predicted_peak_distance) == len(predicted_peak_indicator))

    positive_distance_indices = predicted_peak_distance < args.window

    positive_peak_distances = predicted_peak_distance[
        positive_distance_indices]
    postive_predicted_peak_indicator = predicted_peak_indicator[
        positive_distance_indices]

    print('Neg Peaks: {} Pos Peaks: {}'.format(
        len(predicted_peak_distance) - len(positive_peak_distances),
        len(positive_peak_distances)))

    gci_locations = [
        indices[i, d] for i, d in enumerate(positive_peak_distances)
    ]

    locations_true = np.nonzero(actual_gci_locations)[0]
    xaxes = np.zeros(len(actual_gci_locations))
    xaxes[locations_true] = 1

    if __debug__:
        ground_truth = np.row_stack((np.arange(len(actual_gci_locations)),
                                     xaxes))
        predicted_truth = np.row_stack((gci_locations,
                                        postive_predicted_peak_indicator))
        os.makedirs(args.prediction_dir, exist_ok=True)

        np.save(
            os.path.join(args.prediction_dir, 'ground_truth'), ground_truth)
        np.save(
            os.path.join(args.prediction_dir, 'predicted'), predicted_truth)

    plt.scatter(
        gci_locations,
        postive_predicted_peak_indicator,
        color='b',
        label='Predicted GCI')
    plt.plot(
        np.arange(len(actual_gci_locations)),
        xaxes,
        color='r',
        label='Actual GCI')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
