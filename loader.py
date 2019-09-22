import os
from glob import glob

import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset, Dataset
import warnings
from utils import strided_app
import sys


def create_dataloader(batch_size,
                      speechfolder,
                      peakfolder,
                      window,
                      stride,
                      pin_memory=True):
    def _create_dataset(speechfolder, peakfolder, window, stride):
        speechfiles = sorted(glob(os.path.join(speechfolder, '*.npy')))
        peakfiles = sorted(glob(os.path.join(peakfolder, '*.npy')))

        speech_data = [np.load(f) for f in speechfiles]
        peak_data = [np.load(f) for f in peakfiles]

        speech_data = np.concatenate(speech_data)
        peak_data = np.concatenate(peak_data)

        speech_windowed_data = strided_app(speech_data, window, stride)
        peak_windowed_data = strided_app(peak_data, window, stride)

        peak_distance = np.array([
            np.nonzero(t)[0][0] if len(np.nonzero(t)[0]) != 0 else -1
            for t in peak_windowed_data
        ])

        peak_indicator = (peak_distance != -1) * 1.0

        return speech_windowed_data, peak_distance, peak_indicator

    speech_windowed_data, peak_distance, peak_indicator = _create_dataset(
        speechfolder, peakfolder, window, stride)
    peak_dataset = np.column_stack((peak_distance, peak_indicator))
    dataset = TensorDataset(
        th.from_numpy(
            np.expand_dims(speech_windowed_data, 1).astype(np.float32)),
        th.from_numpy(peak_dataset.astype(np.float32)))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        drop_last=True,
        shuffle=True)
    return dataloader


def create_train_validate_test_data(speechfolder,
                                    peakfolder,
                                    model_folder,
                                    window=80,
                                    stride=1,
                                    batch_size=512,
                                    split=0.7,
                                    numfiles=-1,
                                    pin_memory=True,
                                    symlink=False,
                                    load_test=True):
    def _create_dataset(speechfiles, peakfiles, window, stride):

        speech_data = [np.load(f) for f in speechfiles]
        peak_data = [np.load(f) for f in peakfiles]

        speech_data = np.concatenate(speech_data)
        peak_data = np.concatenate(peak_data)

        speech_windowed_data = strided_app(speech_data, window, stride)
        peak_windowed_data = strided_app(peak_data, window, stride)

        peak_distance = np.array([
            np.nonzero(t)[0][0] if len(np.nonzero(t)[0]) != 0 else -1
            for t in peak_windowed_data
        ])

        peak_indicator = (peak_distance != -1) * 1.0

        return speech_windowed_data, peak_distance, peak_indicator

    def _transform_dataset(dataset):
        if dataset is None:
            return None
        speech_windowed_data, peak_distance, peak_indicator = dataset
        peak_dataset = np.column_stack(
            (peak_distance, peak_indicator))
        dataset = TensorDataset(
            th.from_numpy(
                np.expand_dims(speech_windowed_data, 1).astype(np.float32)),
            th.from_numpy(peak_dataset.astype(np.float32)))

        return dataset

    if isinstance(batch_size, int):
        batch_size = {k: batch_size for k in ['train', 'validate', 'test']}
    elif isinstance(batch_size, dict):
        batch_size.setdefault('train', 512)
        batch_size.setdefault('validate', 512)
        batch_size.setdefault('test', 512)
    else:
        raise ValueError(
            'Incorrect Batch Size Argument, Must be an integer or dict')

    if isinstance(split, float):
        split = {'train': split, 'validate': 1 - split, 'test': 0}
    elif not isinstance(split, dict):
        raise ValueError('Incorrect split Argument, Must be a float or dict')

    if isinstance(stride, int):
        stride = {k: stride for k in ['train', 'validate', 'test']}
    elif isinstance(stride, dict):
        stride.setdefault('train', 1)
        stride.setdefault('validate', 1)
        stride.setdefault('test', 1)
    else:
        raise ValueError('stride must be a dict or integer')

    speechfiles = sorted(glob(os.path.join(speechfolder, '*.npy')))
    speechfiles = [os.path.basename(f) for f in speechfiles]
    peakfiles = sorted(glob(os.path.join(peakfolder, '*.npy')))
    peakfiles = [os.path.basename(f) for f in peakfiles]

    speaker_a_files = [(f1, f2) for f1, f2 in zip(speechfiles, peakfiles)
                       if f1.split('_')[-1] == f2.split('_')[-1]
                       and f1.split('_')[-1][0] == 'a']
    speaker_b_files = [(f1, f2) for f1, f2 in zip(speechfiles, peakfiles)
                       if f1.split('_')[-1] == f2.split('_')[-1]
                       and f1.split('_')[-1][0] == 'b']

    if len(speechfiles) != len(peakfiles):
        warnings.warn('Unequal Speech and Peaks Files, Speech {} Peaks {} Speaker A {} Speaker B {}'.format(
            *map(len, [speechfiles, peakfiles, speaker_a_files, speaker_b_files])))

    if numfiles != -1:
        if isinstance(numfiles, int):
            speaker_a_files = speaker_a_files[:numfiles]
            speaker_b_files = speaker_b_files[:numfiles]
            numfiles = {'a': numfiles, 'b': numfiles}
        else:
            speaker_a_files = speaker_a_files[:numfiles['a']]
            speaker_b_files = speaker_b_files[:numfiles['b']]
    else:
        numfiles = {'a': len(speaker_a_files), 'b': len(speaker_b_files)}

    speaker_a_indices = (
        int(split['train'] * numfiles['a']),
        int((split['train'] + split['validate']) * numfiles['a']),
        int((split['train'] + split['validate'] + split['test']) *
            numfiles['a']))

    speaker_b_indices = (
        int(split['train'] * numfiles['b']),
        int((split['train'] + split['validate']) * numfiles['b']),
        int((split['train'] + split['validate'] + split['test']) *
            numfiles['b']))

    speaker_a_split_files = [
        speaker_a_files[:speaker_a_indices[0]],
        speaker_a_files[speaker_a_indices[0]:speaker_a_indices[1]],
        speaker_a_files[speaker_a_indices[1]:speaker_a_indices[2]]
    ]

    speaker_b_split_files = [
        speaker_b_files[:speaker_b_indices[0]],
        speaker_b_files[speaker_b_indices[0]:speaker_b_indices[1]],
        speaker_b_files[speaker_b_indices[1]:speaker_b_indices[2]]
    ]

    train_files = speaker_a_split_files[0] + speaker_b_split_files[0]
    validate_files = speaker_a_split_files[1] + speaker_b_split_files[1]
    test_files = speaker_a_split_files[2] + speaker_b_split_files[2]

    if __debug__:
        print(
            'Training Files: {} Validation Files: {} Testing Files: {} Speaker A Net: {} Speaker B Net: {}'.format(
                len(train_files), len(validate_files), len(test_files), len(speaker_a_files), len(speaker_b_files)))

    train_speech_files = [
        os.path.join(speechfolder, t[0]) for t in train_files
    ]
    train_peak_files = [os.path.join(peakfolder, t[1]) for t in train_files]

    validate_speech_files = [
        os.path.join(speechfolder, t[0]) for t in validate_files
    ]
    validate_peak_files = [
        os.path.join(peakfolder, t[1]) for t in validate_files
    ]

    test_speech_files = [os.path.join(speechfolder, t[0]) for t in test_files]
    test_peak_files = [os.path.join(peakfolder, t[1]) for t in test_files]

    train_dataset = _transform_dataset(
        _create_dataset(train_speech_files, train_peak_files, window,
                        stride['train']))
    validate_dataset = _transform_dataset(
        _create_dataset(validate_speech_files, validate_peak_files, window,
                        stride['validate']))

    if len(test_speech_files) > 0 and load_test:
        test_dataset = _transform_dataset(
            _create_dataset(test_speech_files, test_peak_files, window,
                            stride['test']))
    else:
        test_dataset = None

    if symlink:
        try:
            os.makedirs(model_folder, exist_ok=True)
            train_speech_dir = os.path.join(model_folder, 'train/speech')
            train_peaks_dir = os.path.join(model_folder, 'train/peaks')
            validate_speech_dir = os.path.join(model_folder, 'validate/speech')
            validate_peaks_dir = os.path.join(model_folder, 'validate/peaks')
            test_speech_dir = os.path.join(model_folder, 'test/speech')
            test_peaks_dir = os.path.join(model_folder, 'test/peaks')
            os.makedirs(train_speech_dir)
            os.makedirs(train_peaks_dir)
            os.makedirs(validate_speech_dir)
            os.makedirs(validate_peaks_dir)
            os.makedirs(test_speech_dir)
            os.makedirs(test_peaks_dir)

            for files, dest_dir in zip([train_speech_files, train_peak_files, validate_speech_files, validate_peak_files, test_speech_files, test_peak_files],
                                       [train_speech_dir, train_peaks_dir, validate_speech_dir, validate_peaks_dir, test_speech_dir, test_peaks_dir]):
                for file in files:
                    src = os.path.join(os.getcwd(), file)
                    basename = os.path.basename(src)
                    dst = os.path.join(os.getcwd(), dest_dir, basename)
                    os.symlink(src, dst)
        except OSError as ose:
            print('Symlink Directories already exist. Please remove old data first')
            print(ose)

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size['train'],
            drop_last=True,
            pin_memory=pin_memory,
            shuffle=True)
    else:
        train_loader = None

    validate_loader = DataLoader(
        validate_dataset,
        batch_size=batch_size['validate'],
        drop_last=True,
        pin_memory=pin_memory,
        shuffle=True)

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size['test'],
            drop_last=True,
            pin_memory=pin_memory,
            shuffle=True)
    else:
        test_loader = None

    return train_loader, validate_loader, test_loader


if __name__ == "__main__":
    train_data, validate_data, test_data = create_train_validate_test_data(
        '0', 'peaks', split={'train': 0.01, 'validate': 0.01, 'test': 0.01}, symlink=True)
