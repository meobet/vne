import numpy as np

import torch
from torch.autograd import Variable


# df/dx - f must be a function or module
def gradient(f, x):
    x = Variable(x.data, requires_grad=True)
    f(x).backward()
    return x.grad


def to_binary(index, output_size, use_cuda=False):
    zeros = torch.zeros(output_size)
    if use_cuda:
        zeros = zeros.cuda()
    return zeros.scatter_(1, index, 1.)


def variable(model, data):
    def _wrap(model, data):
        if model.use_cuda:
            return tuple(Variable(tensor.cuda()) for tensor in data)
        else:
            return tuple(Variable(tensor) for tensor in data)
    if type(data) is tuple or type(data) is list:
        return _wrap(model, data)
    else:
        return _wrap(model, (data, ))[0]


def tensor(model, data):
    def _unwrap(model, data):
        if model.use_cuda:
            return tuple(variable.cpu().data for variable in data)
        else:
            return tuple(variable.data for variable in data)
    if type(data) is tuple or type(data) is list:
        return _unwrap(model, data)
    else:
        return _unwrap(model, (data, ))[0]


def numpy(model, data):
    return tensor(model, data).numpy()


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, decay=0.1, lr_decay_epoch=7):
    # Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
    lr = init_lr * (decay ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr