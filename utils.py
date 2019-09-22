import numpy as np
import glob
import os
import torch.nn as nn
import sys
import re
from natsort import natsorted
import shutil


class Tee():
    def __init__(self, tee_filename):
        try:
            self.tee_fil = open(tee_filename, "w")
        except IOError as ioe:
            raise IOError("Caught Exception: {}".format(repr(ioe)))
        except Exception as e:
            raise Exception("Caught Exception: {}".format(repr(e)))

    def write(self, s):
        sys.stdout.write(s)
        self.tee_fil.write(s)

    def writeln(self, s):
        self.write(s + '\n')
        self.tee_fil.flush()

    def close(self):
        self.tee_fil.close()


class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    return a[S * np.arange(nrows)[:, None] + np.arange(L)]


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def class_imbalance(arr):
    arr = arr.ravel()
    assert (
        len(arr) == np.count_nonzero(arr == 0) + np.count_nonzero(arr == 1))
    return np.count_nonzero(arr) / len(arr)


def get_unstrided_data():
    prefix = 'data'
    train_features_directory = os.path.join(prefix, 'speech', '*.npy')
    train_labels_directory = os.path.join(prefix, 'peaks', '*.npy')

    feature_files = sorted(glob.glob(train_features_directory))
    label_files = sorted(glob.glob(train_labels_directory))

    x_data = np.concatenate([np.load(f) for f in feature_files])
    y_data = np.concatenate([np.load(f) for f in label_files])

    assert (len(x_data) == len(y_data))
    N = len(x_data)
    split_point = int(0.8 * N)

    x_train = x_data[:split_point]
    y_train = y_data[:split_point]
    x_test = x_data[split_point:]
    y_test = y_data[split_point:]

    return (x_train, y_train), (x_test, y_test)


def get_model_dir(directory=None, model_prefix='Model'):
    if directory is None:
        directory = os.getcwd()
    dirs = '\n'.join(os.listdir(directory))
    pattern = re.compile('\s*model\d+', re.IGNORECASE)
    matches = pattern.findall(dirs)
    for i in range(len(matches)):
        matches[i] = matches[i].strip()
    matches = natsorted(matches, key=lambda x: x.lower())
    try:
        mind = matches[-1].lower().find('model')
        curmodel = int(matches[-1][mind+5:])
    except (ValueError, TypeError, IndexError) as v:
        curmodel = 0
    curmodel += 1
    return ''.join([model_prefix, str(curmodel)])


def copy_files(copydir, includefiles=[], excludefiles=[]):
    defaultfiles = set(['cluster.py', 'loader.py', 'loss.py', 'metrics.py', 'model_summarize.py', 'predict.py', 'main.py',
                        'saver.py', 'torch_utils.py', 'utils.py', 'trainer.py', 'vfc.py', 'visualize_cluster.py', 'visualize.py', 'test.py'])
    includefiles = set(includefiles)
    excludefiles = set(excludefiles)
    netfiles = defaultfiles.union(includefiles)
    netfiles.difference_update(excludefiles)

    for file in netfiles:
        if os.path.isfile(file):
            shutil.copy(file, copydir)
