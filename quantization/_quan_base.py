"""
    Quantized modules: the base class
"""
import torch
import torch.nn as nn
from enum import Enum
from torch.nn.parameter import Parameter

__all__ = ['Qmodes', '_Conv2dQ', '_LinearQ', '_ActQ']


class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2

class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        # self.nbits = kwargs_q['nbits']
        self.nbits = Parameter(torch.tensor([float(kwargs_q['nbits'])]), requires_grad = kwargs_q['mixpre'])
        # self.nbits = Parameter(torch.tensor([4.]))
        if kwargs_q['nbits'] < 0:
            self.register_parameter('alpha', None)
            return
        self.q_mode = kwargs_q['mode']
        self.learned = kwargs_q['learned']
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha = Parameter(torch.Tensor(out_channels), requires_grad=self.learned)
        else:  # layer-wise quantization
            self.alpha = Parameter(torch.zeros(7), requires_grad=self.learned)
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_Conv2dQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)

class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs_q):
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        # self.nbits = kwargs_q['nbits']
        self.nbits = Parameter(torch.tensor([float(kwargs_q['nbits'])]), requires_grad = kwargs_q['mixpre'])
        self.learned = kwargs_q['learned']
        if kwargs_q['nbits'] < 0:
            self.register_parameter('alpha', None)
            return
        self.alpha = Parameter(torch.zeros(7), requires_grad=self.learned)

        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_LinearQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)


class _ActQ(nn.Module):
    def __init__(self, **kwargs_q):
        super(_ActQ, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        # self.nbits = kwargs_q['nbits']
        self.nbits = Parameter(torch.tensor([float(kwargs_q['nbits'])]), requires_grad = kwargs_q['mixpre'])
        self.learned = kwargs_q['learned']
        if kwargs_q['nbits'] < 0:
            self.register_parameter('alpha', None)
            return
        self.signed = kwargs_q['signed']
        self.offset = kwargs_q['offset']
        self.dim = kwargs_q['dim']
        self.alpha = Parameter(torch.zeros(7), requires_grad=self.learned)
        if self.offset:
            self.beta = Parameter(torch.zeros(7), requires_grad=self.learned)
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        # s_prefix = super(_ActQ, self).extra_repr()
        if self.alpha is None:
            return 'fake'
        return '{}'.format(self.kwargs_q)

def get_default_kwargs_q(kwargs_q, layer_type):
    default = {
        'nbits': 8
    }
    if isinstance(layer_type, _Conv2dQ):
        default.update({
            'mode': Qmodes.layer_wise})
    elif isinstance(layer_type, _LinearQ):
        pass
    elif isinstance(layer_type, _ActQ):
        default.update({
            'signed': True})
    else:
        assert NotImplementedError
        return
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q


class _MultiHeadActQ(nn.Module):
    def __init__(self, **kwargs_q):
        super(_MultiHeadActQ, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        # self.nbits = kwargs_q['nbits']
        self.num_head = kwargs_q['num_head']
        self.nbits = Parameter(torch.ones(self.num_head) * kwargs_q['nbits'])
        # self.nbits = Parameter(torch.tensor([float(kwargs_q['nbits'])]))
        self.learned = kwargs_q['learned']
        if kwargs_q['nbits'] < 0:
            self.register_parameter('alpha', None)
            return
        self.signed = kwargs_q['signed']
        self.offset = kwargs_q['offset']
        self.dim = kwargs_q['dim']
        self.alpha = Parameter(torch.zeros(7), requires_grad=self.learned)
        if self.offset:
            self.beta = Parameter(torch.zeros(7), requires_grad=self.learned)
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        # s_prefix = super(_ActQ, self).extra_repr()
        if self.alpha is None:
            return 'fake'
        return '{}'.format(self.kwargs_q)

class _MultiHeadLinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs_q):
        super(_MultiHeadLinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        # self.nbits = kwargs_q['nbits']
        self.num_head = kwargs_q['num_head']
        self.nbits = Parameter(torch.ones(self.num_head) * kwargs_q['nbits'])
        # self.nbits = Parameter(torch.tensor([float(kwargs_q['nbits'])]))
        self.learned = kwargs_q['learned']
        if kwargs_q['nbits'] < 0:
            self.register_parameter('alpha', None)
            return
        self.alpha = Parameter(torch.zeros(7), requires_grad=self.learned)

        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_MultiHeadLinearQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)

def get_default_kwargs_q(kwargs_q, layer_type):
    default = {
        'nbits': 8
    }
    if isinstance(layer_type, _Conv2dQ):
        default.update({
            'mode': Qmodes.layer_wise})
    elif isinstance(layer_type, _LinearQ):
        pass
    elif isinstance(layer_type, _ActQ):
        default.update({
            'signed': True})
    else:
        assert NotImplementedError
        return
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q