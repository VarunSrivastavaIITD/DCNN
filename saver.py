import os
import torch as th
import torch.nn as nn
import torch.optim as optim
from time import localtime, strftime


class Saver:
    def __init__(self, directory: str = 'pytorch_model') -> None:
        self.directory = directory

    def save_checkpoint(self,
                        state,
                        file_name: str = 'pytorch_model.pt',
                        append_time=True):
        os.makedirs(self.directory, exist_ok=True)
        timestamp = strftime("%Y_%m_%d__%H_%M_%S", localtime())
        filebasename, fileext = file_name.split('.')
        if append_time:
            filepath = os.path.join(self.directory, '_'.join(
                [filebasename, '.'.join([timestamp, fileext])]))
        else:
            filepath = os.path.join(self.directory, file_name)
        if isinstance(state, nn.Module):
            checkpoint = {'model_dict': state.state_dict()}
            th.save(checkpoint, filepath)
        elif isinstance(state, dict):
            th.save(state, filepath)
        else:
            raise TypeError('state must be a nn.Module or dict')

    def load_checkpoint(self,
                        model: nn.Module,
                        optimizer: optim.Optimizer = None,
                        file_name: str = 'pytorch_model.pt'):
        filepath = os.path.join(self.directory, file_name)
        checkpoint: dict = th.load(
            filepath, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_dict'])

        hyperparam_dict = {
            k: v
            for k, v in checkpoint.items()
            if k != 'model_dict' or k != 'optimizer_dict'
        }

        return model, optimizer, hyperparam_dict

    def create_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                          hyperparam_dict):
        model_dict = model.state_dict()
        optimizer_dict = optimizer.state_dict()

        state_dict = {
            'model_dict': model_dict,
            'optimizer_dict': optimizer_dict,
            'timestamp': strftime('%l:%M%p GMT%z on %b %d, %Y', localtime())
        }
        checkpoint = {**state_dict, **hyperparam_dict}

        return checkpoint
