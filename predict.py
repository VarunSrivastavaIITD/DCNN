from glob import glob
import warnings
import os
import numpy as np
from utils import strided_app
from metrics import corrected_naylor_metrics
import torch as th
from torch_utils import to_variable
import torch.nn.functional as F
from cluster import cluster


def create_dataset(speechfolder,
                   peaksfolder,
                   window,
                   stride,
                   file_slice=slice(0, 10)):
    speechfiles = sorted(glob(os.path.join(speechfolder, '*.npy')))[file_slice]
    peakfiles = sorted(glob(os.path.join(peaksfolder, '*.npy')))[file_slice]

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


def get_optimal_params(model, speechfolder,
                       peaksfolder,
                       window,
                       stride,
                       filespan,
                       numfiles=10,
                       use_cuda=True,
                       thlist=[0.6, 0.7],
                       spblist=[25],
                       hctlist=[15, 20, 25]):
    model.eval()
    if not use_cuda:
        model.cpu()
    filespan = 10
    numfiles = min([len(glob(os.path.join(speechfolder, '*.npy'))), numfiles])
    idr_dict = {'idr': 0, 'thr': -1, 'spb': -1,
                'hct': -1, 'mr': -1, 'far': -1, 'se': -1}
    far_dict = {'far': 1, 'thr': -1, 'spb': -1,
                'hct': -1, 'mr': -1, 'idr': -1, 'se': -1}
    mr_dict = {'mr': 1, 'thr': -1, 'spb': -1,
               'hct': -1, 'idr': -1, 'far': -1, 'se': -1}

    for thr in thlist:
        for spb in spblist:
            for hct in hctlist:
                metrics_list = []

                for i in range(0, numfiles, filespan):
                    if (i + filespan) > numfiles:
                        break
                    speech_windowed_data, peak_distance, peak_indicator, indices, actual_gci_locations = create_dataset(
                        speechfolder, peaksfolder, window, stride, slice(i, i + filespan))

                    input = to_variable(
                        th.from_numpy(
                            np.expand_dims(speech_windowed_data, 1).astype(np.float32)),
                        use_cuda, True)

                    with warnings.catch_warnings():
                        prediction = model(input)

                    predicted_peak_indicator = F.sigmoid(
                        prediction[:, 1]).cpu().data.numpy()
                    predicted_peak_distance = (prediction[:, 0]).cpu().data.numpy().astype(
                        np.int32)

                    predicted_peak_indicator_indices = predicted_peak_indicator > 0

                    predicted_peak_indicator = predicted_peak_indicator[
                        predicted_peak_indicator_indices].ravel()
                    predicted_peak_distance = predicted_peak_distance[
                        predicted_peak_indicator_indices].ravel()
                    indices = indices[predicted_peak_indicator_indices]

                    positive_distance_indices = predicted_peak_distance < window

                    positive_peak_distances = predicted_peak_distance[
                        positive_distance_indices]
                    postive_predicted_peak_indicator = predicted_peak_indicator[
                        positive_distance_indices]

                    gci_locations = [
                        indices[i, d] for i, d in enumerate(positive_peak_distances)
                    ]

                    locations_true = np.nonzero(actual_gci_locations)[0]
                    xaxes = np.zeros(len(actual_gci_locations))
                    xaxes[locations_true] = 1

                    ground_truth = np.row_stack((np.arange(len(actual_gci_locations)),
                                                 xaxes))
                    predicted_truth = np.row_stack((gci_locations,
                                                    postive_predicted_peak_indicator))

                    gx = ground_truth[0, :]
                    gy = ground_truth[1, :]

                    px = predicted_truth[0, :]
                    py = predicted_truth[1, :]

                    fs = 16000

                    gci = np.array(
                        cluster(
                            px,
                            py,
                            threshold=thr,
                            samples_per_bin=spb,
                            histogram_count_threshold=hct))
                    predicted_gci_time = gci / fs
                    target_gci_time = np.nonzero(gy)[0] / fs

                    gci = np.round(gci).astype(np.int64)
                    gcilocs = np.zeros_like(gx)
                    gcilocs[gci] = 1

                    metric = corrected_naylor_metrics(
                        target_gci_time, predicted_gci_time)
                    metrics_list.append(metric)

                idr = np.mean([
                    v for m in metrics_list for k, v in m.items()
                    if k == 'identification_rate'
                ])
                mr = np.mean(
                    [v for m in metrics_list for k, v in m.items() if k == 'miss_rate'])
                far = np.mean([
                    v for m in metrics_list for k, v in m.items()
                    if k == 'false_alarm_rate'
                ])
                se = np.mean([
                    v for m in metrics_list for k, v in m.items()
                    if k == 'identification_accuracy'
                ])

                if idr_dict['idr'] < idr:
                    idr_dict['idr'] = idr
                    idr_dict['thr'] = thr
                    idr_dict['spb'] = spb
                    idr_dict['hct'] = hct
                    idr_dict['far'] = far
                    idr_dict['mr'] = mr
                    idr_dict['se'] = se
                if mr_dict['mr'] > mr:
                    mr_dict['idr'] = idr
                    mr_dict['thr'] = thr
                    mr_dict['spb'] = spb
                    mr_dict['hct'] = hct
                    mr_dict['far'] = far
                    mr_dict['mr'] = mr
                    mr_dict['se'] = se
                if far_dict['far'] > far:
                    far_dict['idr'] = idr
                    far_dict['thr'] = thr
                    far_dict['spb'] = spb
                    far_dict['hct'] = hct
                    far_dict['far'] = far
                    far_dict['mr'] = mr
                    far_dict['se'] = se
    return idr_dict, mr_dict, far_dict


def model_naylor_metrics(model, speechfolder,
                         peaksfolder,
                         window,
                         stride,
                         filespan=10,
                         use_cuda=True,
                         param_dict={'thr': 0.7, 'spb': 25, 'hct': 20}):
    param_dict.setdefault('thr', 0.7)
    param_dict.setdefault('spb', 25)
    param_dict.setdefault('hct', 20)
    thr = param_dict['thr']
    spb = param_dict['spb']
    hct = param_dict['hct']
    model.eval()
    if not use_cuda:
        model.cpu()
    filespan = 10
    numfiles = len(glob(os.path.join(speechfolder, '*.npy')))

    metrics_list = []

    for i in range(0, numfiles, filespan):
        if (i + filespan) > numfiles:
            break
        speech_windowed_data, peak_distance, peak_indicator, indices, actual_gci_locations = create_dataset(
            speechfolder, peaksfolder, window, stride, slice(i, i + filespan))

        input = to_variable(
            th.from_numpy(
                np.expand_dims(speech_windowed_data, 1).astype(np.float32)),
            use_cuda, True)

        with warnings.catch_warnings():
            prediction = model(input)

        predicted_peak_indicator = F.sigmoid(
            prediction[:, 1]).cpu().data.numpy()
        predicted_peak_distance = (prediction[:, 0]).cpu().data.numpy().astype(
            np.int32)

        predicted_peak_indicator_indices = predicted_peak_indicator > 0

        predicted_peak_indicator = predicted_peak_indicator[
            predicted_peak_indicator_indices].ravel()
        predicted_peak_distance = predicted_peak_distance[
            predicted_peak_indicator_indices].ravel()
        indices = indices[predicted_peak_indicator_indices]

        positive_distance_indices = predicted_peak_distance < window

        positive_peak_distances = predicted_peak_distance[
            positive_distance_indices]
        postive_predicted_peak_indicator = predicted_peak_indicator[
            positive_distance_indices]

        gci_locations = [
            indices[i, d] for i, d in enumerate(positive_peak_distances)
        ]

        locations_true = np.nonzero(actual_gci_locations)[0]
        xaxes = np.zeros(len(actual_gci_locations))
        xaxes[locations_true] = 1

        ground_truth = np.row_stack((np.arange(len(actual_gci_locations)),
                                     xaxes))
        predicted_truth = np.row_stack((gci_locations,
                                        postive_predicted_peak_indicator))

        gx = ground_truth[0, :]
        gy = ground_truth[1, :]

        px = predicted_truth[0, :]
        py = predicted_truth[1, :]

        fs = 16000

        gci = np.array(
            cluster(
                px,
                py,
                threshold=thr,
                samples_per_bin=spb,
                histogram_count_threshold=hct))
        predicted_gci_time = gci / fs
        target_gci_time = np.nonzero(gy)[0] / fs

        gci = np.round(gci).astype(np.int64)
        gcilocs = np.zeros_like(gx)
        gcilocs[gci] = 1

        metric = corrected_naylor_metrics(
            target_gci_time, predicted_gci_time)
        metrics_list.append(metric)

    idr = np.mean([
        v for m in metrics_list for k, v in m.items()
        if k == 'identification_rate'
    ])
    mr = np.mean(
        [v for m in metrics_list for k, v in m.items() if k == 'miss_rate'])
    far = np.mean([
        v for m in metrics_list for k, v in m.items()
        if k == 'false_alarm_rate'
    ])
    se = np.mean([
        v for m in metrics_list for k, v in m.items()
        if k == 'identification_accuracy'
    ])

    return idr, mr, far, se
