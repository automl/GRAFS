import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class F_gpt2_openweb_gelu_1_1(nn.Module):
    """
    seed 1, layer 1
    """

    def __init__(self):
        super(F_gpt2_openweb_gelu_1_1, self).__init__()

    def forward(self, x):
        return F.silu(F.silu(x) + F.gelu(x)) + F.gelu(x)


class F_gpt2_openweb_gelu_2_1(nn.Module):
    """
    seed 2, layer 1
    """

    def __init__(self):
        super(F_gpt2_openweb_gelu_2_1, self).__init__()

    def forward(self, x):
        return F.gelu(F.gelu(x)) + F.gelu(x)


class F_gpt2_openweb_gelu_2_3(nn.Module):
    """
    seed 2, layer 3
    """

    def __init__(self):
        super(F_gpt2_openweb_gelu_2_3, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.sigmoid(F.gelu(x)) * F.gelu(x)) * F.gelu(x)


class F_gpt2_openweb_gelu_2_5(nn.Module):
    """
    seed 2, layer 5
    """

    def __init__(self):
        super(F_gpt2_openweb_gelu_2_5, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.leaky_relu(F.gelu(x))) * F.gelu(x)


class F_nanoGPT_tinystories_gelu_0(nn.Module):
    """
    seed 2, layer 5
    """

    def __init__(self):
        super(F_nanoGPT_tinystories_gelu_0, self).__init__()

    def forward(self, x):
        return F.silu(F.sigmoid(F.relu(x)) * F.gelu(x)) * F.silu(x)



class F_nanoGPT_tinystories_gelu_1(nn.Module):
    """
    seed 2, layer 5
    """

    def __init__(self):
        super(F_nanoGPT_tinystories_gelu_1, self).__init__()

    def forward(self, x):
        return 0.3994 * F.gelu(0.4413 * torch.square(x) + 0.5587 * F.gelu(x)) + 0.6006 * F.silu(x)


class F_nanoGPT_tinystories_gelu_2(nn.Module):
    """
    seed 2, layer 5
    """

    def __init__(self):
        super(F_nanoGPT_tinystories_gelu_2, self).__init__()

    def forward(self, x):
        return 0.431 * F.silu(F.sigmoid(F.silu(x)) * torch.pow(x, 2)) + 0.5681 * F.silu(x)


class F_nanoGPT_tinystories_gelu_3(nn.Module):
    """
    seed 2, layer 5
    """

    def __init__(self):
        super(F_nanoGPT_tinystories_gelu_3, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.gelu(torch.pow(F.silu(x), 2)))
