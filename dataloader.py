import torch as th
import numpy as np
from peakdetect import peakdetect
from torch.utils.data import Dataset, DataLoader, Subset


class AudioDataset(Dataset):
    def __init__(self, speechfile, eggfile, window, stride):
        self.speechfile = speechfile
        self.eggfile = eggfile
        self.window = window
        self.stride = stride
        self.speech = np.load(self.speechfile)
        self.egg = np.load(self.eggfile)

    @staticmethod
    def process_peaks(peak_locations, sig_size):
        diff_fin = np.diff(peak_locations)
        threshold = 50
        thresholded_diff = (diff_fin >= threshold) * 1.0
        final_diff = np.append(thresholded_diff, 1) * peak_locations

        fin = final_diff.astype(np.int)
        fin = fin[np.nonzero(fin)]

        ground_truth = np.zeros(sig_size)
        ground_truth[fin] = 1
        return ground_truth

    @staticmethod
    def getregions(wav):
        degg = np.insert(np.diff(wav), 0, 0)
        abs_degg = np.abs(degg)

        largest_ind = np.argpartition(abs_degg, -100)[-100:]
        thresh = -1 / 6 * np.mean(abs_degg[largest_ind])

        # Select points with value less than threshold, to detect points near gci showing the voiced regions
        out = degg < thresh
        samples = np.nonzero(out * 1.0)
        locn = np.array(samples).reshape(-1)
        # insert very high value, since it has to be the last voiced regional point
        dist = np.append(np.diff(locn), 10000)
        # Two voice regions have to be atleast 1000 samples apart
        dec = np.array(np.nonzero(1.0 * (dist > 1000))).reshape(-1)
        end = locn[dec]  # Ending positions of Voiced regions
        # one point after in the thresholded list locations will give starings except at the  last point
        start = locn[dec[:-1] + 1]
        # Insert the first thresholded point as starting of first voiced regions
        start = np.insert(start, 0, locn[0])

        return start, end, degg

    @staticmethod
    def getpeaks(data, one_hot=False):
        degg = np.insert(np.diff(data), 0, 0)  # To preserve no. of inputs
        out = np.array(peakdetect(degg, lookahead=5))
        out = np.array(out[1])[:, 0]
        out = out.astype(np.int)

        abs_degg = np.abs(degg)

        largest_ind = np.argpartition(abs_degg, -100)[-100:]
        thresh = -1 / 6 * np.mean(abs_degg[largest_ind])

        # Apply Threshold
        dec = degg[out] <= thresh
        fin = out[np.nonzero(1 * dec)]

        diff_fin = np.diff(fin)
        threshold = 50
        thresholded_diff = (diff_fin >= threshold) * 1.0
        final_diff = np.insert(thresholded_diff, len(thresholded_diff) - 1, 1) * fin

        fin = final_diff.astype(np.int)

        if one_hot:
            ground_truth = np.zeros(len(degg))
            ground_truth[fin] = 1
            return fin, ground_truth, degg

        return fin, degg[fin], degg

    @staticmethod
    def normalize(data):
        import pandas as pd

        s = pd.Series(data)
        maxmean = s.nlargest(200).median()
        minmean = s.nsmallest(200).median()

        data = 2 * ((data - minmean) / (maxmean - minmean)) - 1

        # data = (data - np.mean(data)) / np.std(data)

        data = np.clip(data, -1, 1)
        return data

    def __len__(self):
        return len(self.speech.files)

    def __getitem__(self, idx):
        shifted_speech = self.speech[self.speech.files[idx]]
        egg = self.egg[self.egg.files[idx]]

        _, peaks, _ = self.getpeaks(egg, True)
        start, end, _ = self.getregions(egg)

        final_speech = [np.array(shifted_speech[st:en]) for st, en in zip(start, end)]
        final_peaks = [np.array(peaks[st:en]) for st, en in zip(start, end)]

        speech = np.concatenate(final_speech, axis=0)
        speech = self.normalize(speech)
        temp_peaks = np.concatenate(final_peaks, axis=0)
        peaks = self.process_peaks(np.nonzero(temp_peaks)[0], len(temp_peaks))

        speech_windowed_data = (
            th.from_numpy(speech).unfold(0, self.window, self.stride).type(th.float32)
        )
        peak_windowed_data = th.from_numpy(peaks).unfold(0, self.window, self.stride)
        peak_distance = np.array(
            [
                np.nonzero(t)[0][0] if len(np.nonzero(t)[0]) != 0 else -1
                for t in peak_windowed_data.numpy()
            ]
        )

        peak_indicator = (peak_distance != -1) * 1.0
        peak_dataset = th.from_numpy(
            np.column_stack((peak_distance, peak_indicator))
        ).type(th.float32)

        return speech_windowed_data, peak_dataset


def partial_random_split(dataset, fracs, random_seed=99):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        fracs (sequence): fractions of splits to be produced
    """
    from torch._utils import _accumulate

    lengths = [int(f * len(dataset)) for f in fracs[:-1]]
    lengths.append(int(sum(fracs) * len(dataset)) - sum(lengths))

    if sum(fracs) > 1:
        raise ValueError(
            "Sum of input lengths is greater than the length of the input dataset!"
        )

    prng = np.random.RandomState(random_seed)
    indices = prng.permutation(sum(lengths))
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def create_train_validation_test_split(dataset: Dataset, fracs, **kwargs):
    if len(fracs) != 3:
        raise ValueError(
            "Exactly 3 ratios must be given, {} provided".format(len(fracs))
        )
    trfrac, valfrac, testfrac = fracs
    train_set, val_set, test_set = partial_random_split(
        dataset, [trfrac, valfrac, testfrac]
    )

    return train_set, val_set, test_set


def create_train_validate_test_dataloader(netconfig):
    dataset = AudioDataset(
        netconfig["speech_npz_file"],
        netconfig["egg_npz_file"],
        netconfig["window"],
        netconfig["stride"],
    )
    fracs = [
        netconfig["train_ratio"],
        netconfig["validate_ratio"],
        netconfig["test_ratio"],
    ]
    train_dataset, validate_dataset, test_dataset = create_train_validation_test_split(
        dataset, fracs
    )
    batch_size = netconfig["batch_size"]
    pin_memory = False

    def collate_fn(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]

        data = th.cat(data).unsqueeze_(1)
        target = th.cat(target)

        return [data, target]

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
        )
    else:
        train_loader = None

    validate_loader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=pin_memory,
        shuffle=True,
        collate_fn=collate_fn,
    )

    if test_dataset is not None and netconfig["load_test"]:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            drop_last=True,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
        )
    else:
        test_loader = None

    return train_loader, validate_loader, test_loader


if __name__ == "__main__":
    pass
