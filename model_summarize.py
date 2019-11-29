from collections import OrderedDict
import pandas as pd

import torch
from torch.autograd import Variable
from torch import nn
import copy


def get_names_dict(model):
    """
    Recursive walk to get names including path
    """
    names = {}

    def _get_names(module, parent_name=''):
        for key, module in module.named_children():
            name = parent_name + '.' + key if parent_name else key
            names[name] = module
            if isinstance(module, torch.nn.Module):
                _get_names(module, parent_name=name)

    _get_names(model)
    return names


def torch_summarize_df(input_size,
                       model,
                       weights=False,
                       input_shape=True,
                       nb_trainable=False):
    """
    Summarizes torch model by showing trainable parameters and weights.
    author: wassname
    url: https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7
    license: MIT
    Modified from:
    - https://github.com/pytorch/pytorch/issues/2001#issuecomment-313735757
    - https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7/
    Usage:
        import torchvision.models as models
        model = models.alexnet()
        df = torch_summarize_df(input_size=(3, 224,224), model=model)
        print(df)

        #              name class_name        input_shape       output_shape  nb_params
        # 1     features=>0     Conv2d  (-1, 3, 224, 224)   (-1, 64, 55, 55)      23296
        # 2     features=>1       ReLU   (-1, 64, 55, 55)   (-1, 64, 55, 55)          0
        # ...
    """

    def register_hook(module):
        def hook(module, input, output):
            name = ''
            for key, item in names.items():
                if item == module:
                    name = key

            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = module_idx + 1

            summary[m_key] = OrderedDict()
            summary[m_key]['name'] = name
            summary[m_key]['class_name'] = class_name
            if input_shape:
                summary[m_key]['input_shape'] = (-1, ) + tuple(
                    input[0].size())[1:]
            summary[m_key]['output_shape'] = (-1, ) + tuple(output.size())[1:]
            if weights:
                summary[m_key]['weights'] = list(
                    [tuple(p.size()) for p in module.parameters()])


#             summary[m_key]['trainable'] = any([p.requires_grad for p in module.parameters()])
            if nb_trainable:
                params_trainable = sum([
                    torch.LongTensor(list(p.size())).prod()
                    for p in module.parameters() if p.requires_grad
                ])
                summary[m_key]['nb_trainable'] = params_trainable
            params = sum([
                torch.LongTensor(list(p.size())).prod()
                for p in module.parameters()
            ])
            summary[m_key]['nb_params'] = params

        if not isinstance(module, nn.Sequential) and not isinstance(
                module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # Names are stored in parent and path+name is unique not the name
    # model = copy.deepcopy(model)
    model.eval()
    names = get_names_dict(model)

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size)) for in_size in input_size]
    else:
        x = Variable(torch.rand(1, *input_size))

    if next(model.parameters()).is_cuda:
        x = x.cuda()

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # make dataframe
    df_summary = pd.DataFrame.from_dict(summary, orient='index')

    return df_summary
