import math
import cv2
import numpy as np
import torch as th
from torch import nn
from torch.nn import Conv2d

from .utils import generate_binary_channel, add_binary_channel, set_fixed_type

class ResettableNN(nn.Module):
    """Neural networks that may have internal state that needs to be reset.
    
    For example, NCAs with "auxiliary" activations---channels in the map that are not actually part of the level, but
    used as external memory by the model (therefore, we store them here). Or maybe memory states? Same thing??"""
    def __init__(self, step_size=0.01, **kwargs):
        self.step_size = step_size
        super().__init__()

    def reset(self):
        pass

    def mutate(self):
        set_nograd(self)
        w = get_init_weights(self, init=False, torch=True)

        # Add a random gaussian to the weights, with mean 0 and standard deviation `self.step_size`.
        w += th.randn_like(w) * math.sqrt(self.step_size)

        set_weights(self, w)

class NCA(ResettableNN):
    def __init__(self, n_in_chans, n_actions, n_aux_chan=0, render=False, **kwargs):
        """
        Args:
            render (bool): whether to render the auxiliary channels in order to observe the model's behavior.
        """
        super().__init__(**kwargs)
        self._has_aux = n_aux_chan > 0
        self.n_hid_1 = n_hid_1 = 32
        self.n_aux = n_aux_chan
        self.l1 = Conv2d(n_in_chans + self.n_aux + kwargs.get("binary_channel"), n_hid_1, 3, 1, 1, bias=True)
        self.l2 = Conv2d(n_hid_1, n_hid_1, 1, 1, 0, bias=True)
        self.l3 = Conv2d(n_hid_1, n_actions + self.n_aux, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.last_aux = None
        self.binary_channel = kwargs.get("binary_channel")
        self.apply(init_weights)
        # self._render = render
        self._render = False 
        if self._render:
            cv2.namedWindow("Auxiliary NCA")

    def forward(self, x, **kwargs):
        with th.no_grad():
            if self._has_aux:
                if self.last_aux is None:
                    self.last_aux = th.zeros(size=(1, self.n_aux, *x.shape[-2:]))
                x_in = th.cat([x, self.last_aux], axis=1)
            else:
                x_in = x

            # TODO: change to generalize well for other games
            fixed_tiles = kwargs.get("fixed_tiles")

            if self.binary_channel and fixed_tiles is not None:
                binary_channel = generate_binary_channel(fixed_tiles)
                x_in = add_binary_channel(x_in, binary_channel)

            x = self.l1(x_in)
            x = th.relu(x)
            x = self.l2(x)
            x = th.relu(x)
            x = self.l3(x)
            x = th.sigmoid(x)
            if fixed_tiles is not None: set_fixed_type(x, fixed_tiles)

            if self._has_aux > 0:
                self.last_aux = x[:,-self.n_aux:,:,:]
                x = x[:, :-self.n_aux,:,:]

            if self._render:
                aux = self.last_aux[0].cpu().numpy()
                aux = aux / aux.max()
                im = np.expand_dims(np.vstack(aux), axis=0)
                im = im.transpose(1, 2, 0)
                cv2.imshow("Auxiliary NCA", im)
                cv2.waitKey(1)
        
        return x, False

    def reset(self, init_aux=None):
        self.last_aux = None

def init_siren_weights(m, first_layer=False):
    if first_layer:
        th.nn.init.constant_(m.weight, 30)
        return
    if type(m) == th.nn.Conv2d:
        ws = m.weight.shape
        # number of _inputs
        n = ws[0] * ws[1] * ws[2]
        th.nn.init.uniform_(m.weight, -np.sqrt(6/n), np.sqrt(6/n))
        m.bias.data.fill_(0.01)
    else:
        raise Exception


def init_weights(m):
    if type(m) == th.nn.Linear:
        th.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == th.nn.Conv2d:
        th.nn.init.orthogonal_(m.weight)


def init_play_weights(m):
    if type(m) == th.nn.Linear:
        th.nn.init.xavier_uniform(m.weight, gain=0)
        m.bias.data.fill_(0.00)

    if type(m) == th.nn.Conv2d:
        #       th.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        th.nn.init.constant_(m.weight, 0)


def set_nograd(nn):
    if not hasattr(nn, "parameters"):
        return
    for param in nn.parameters():
        param.requires_grad = False


def get_init_weights(nn, init=True, torch=False):
    """
    Use to get flat vector of weights from PyTorch model
    """
    init_params = []

    for lyr in nn.layers:
        init_params.append(lyr.weight.view(-1))
        if lyr.bias is not None:
            init_params.append(lyr.bias.view(-1))
    if not torch:
        init_params = [p.cpu().numpy() for p in init_params]
        init_params = np.hstack(init_params)
    else:
        init_params = th.cat(init_params)
    if init:
        print("number of initial NN parameters: {}".format(init_params.shape))

    return init_params


def set_weights(nn, weights, algo="CMAME"):
    if algo == "ME":
        # then out nn is contained in the individual
        individual = weights  # I'm sorry mama
        return individual.model
    with th.no_grad():
        n_el = 0

        for layer in nn.layers:
            l_weights = weights[n_el : n_el + layer.weight.numel()]
            n_el += layer.weight.numel()
            l_weights = l_weights.reshape(layer.weight.shape)
            layer.weight = th.nn.Parameter(th.Tensor(np.array(l_weights)))
            layer.weight.requires_grad = False
            if layer.bias is not None:
                n_bias = layer.bias.numel()
                b_weights = weights[n_el : n_el + n_bias]
                n_el += n_bias
                b_weights = b_weights.reshape(layer.bias.shape)
                layer.bias = th.nn.Parameter(th.Tensor(np.array(b_weights)))
                layer.bias.requires_grad = False

    return nn