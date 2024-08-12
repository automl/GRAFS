import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class F_res18_75_cifar100_elu(nn.Module):
    """
    71,184	silu-silu-sig_mul-gelu-silu-beta_mix
    warmstart 5
    """

    def __init__(self):
        super(F_res18_75_cifar100_elu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.relu(x)) * F.elu(x)


class F_vit_tiny_35_cifar10_0_gelu(nn.Module):
    """
    84,096	gelu-gelu-beta_mix-gelu-gelu-beta_mix	0,00639659119769931	-0,129353985190392
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_0_gelu, self).__init__()

    def forward(self, x):
        return F.gelu(x)


class F_vit_tiny_35_cifar10_1_gelu(nn.Module):
    """
    88,488	x_sq-gelu-beta_mix-gelu-silu-beta_mix	-0,235763430595398	-0,408169448375702
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_1_gelu, self).__init__()

    def forward(self, x):
        return 0.3994 * F.gelu(0.4413 * torch.pow(x, 2).clamp(max=10, min=-10) + 0.5587 * F.gelu(x)) + 0.6006 * F.silu(
            x)


class F_vit_tiny_35_cifar10_2_gelu(nn.Module):
    """
    87,664	silu-x_sq-sig_mul-silu-silu-beta_mix	0,109255999326706	-0,274099498987198
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_2_gelu, self).__init__()

    def forward(self, x):
        return 0.4319 * F.silu(torch.sigmoid(F.silu(x)) * torch.pow(x, 2).clamp(max=10, min=-10)) + 0.5681 * F.silu(x)


class F_vit_tiny_35_cifar10_3_gelu(nn.Module):
    """
    85,776	silu-silu-mul-gelu-silu-sig_mul	-0,00260650226846337	-0,205019414424896
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_3_gelu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.gelu(F.silu(x) * F.silu(x))) * F.silu(x)


class F_wresnet28_2_35_cifar10_0_relu(nn.Module):
    """
    97,856	max0-max0-beta_mix-leaky_relu-gelu-sig_mul	0,00564804906025529	-0,0975285843014717
    """

    def __init__(self):
        super(F_wresnet28_2_35_cifar10_0_relu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.relu(x)) * F.gelu(x)


class F_wresnet28_2_35_cifar10_1_relu(nn.Module):
    """
    53	0,0136208364191909	88,9056884906022	96,28	leaky_relu-max0-beta_mix-max0-gelu-sig_mul	-0,0342765040695667	-0,128165870904922
    """

    def __init__(self):
        super(F_wresnet28_2_35_cifar10_1_relu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.relu(x)) * F.gelu(x)


class F_wresnet28_2_35_cifar10_3_relu(nn.Module):
    """
    51	0,0138826808550891	90,5606444098521	96,096	max0-max0-right-max0-gelu-sig_mul	-0,0725536793470383	-0,173576161265373
    """

    def __init__(self):
        super(F_wresnet28_2_35_cifar10_3_relu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.relu(x)) * F.gelu(x)


class F_wresnet28_2_35_cifar10_4_relu(nn.Module):
    """
    49	0,0142797520038522	102,057829594589	95,72	max0-gelu-beta_mix-gelu-gelu-sig_mul	-0,131367057561874	-0,0853490233421326
    """

    def __init__(self):
        super(F_wresnet28_2_35_cifar10_4_relu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.gelu(0.4672 * F.relu(x) + 0.5328 * F.gelu(x))) * F.gelu(x)


class F_res18_35_cifar10_0_relu(nn.Module):
    """
    99	0,00786968200179866	46,7853763250132	98,352	gelu-gelu-max-gelu-gelu-sig_mul	0,00478318566456437	0,0228871498256922
    """

    def __init__(self):
        super(F_res18_35_cifar10_0_relu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.gelu(x)) * F.gelu(x)


class F_res18_35_cifar10_1_relu(nn.Module):
    """
    99	0,00784837578521857	40,1441385956077	98,536	gelu-gelu-min-gelu-gelu-sig_mul	-0,0324761793017387	0,0721149295568466
    """

    def __init__(self):
        super(F_res18_35_cifar10_1_relu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.gelu(x)) * F.gelu(x)


class F_res18_35_cifar10_2_relu(nn.Module):
    """
    99	0,00795807020745475	39,1477069022567	98,528	gelu-gelu-left-leaky_relu-leaky_relu-sig_mul	0,0181609876453877	0,0654356852173805
    """

    def __init__(self):
        super(F_res18_35_cifar10_2_relu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.leaky_relu(F.gelu(x))) * F.leaky_relu(x)


class F_res18_35_cifar10_3_relu(nn.Module):
    """
    98,656	gelu-gelu-sig_mul-max0-max0-left	0,052965197712183	0,0102420253679156
    """

    def __init__(self):
        super(F_res18_35_cifar10_3_relu, self).__init__()

    def forward(self, x):
        return F.relu(F.sigmoid(F.gelu(x)) * F.gelu(x))


class F_res18_35_cifar10_4_relu(nn.Module):
    """
    82 98,424	max0-max0-beta_mix-gelu-max0-sig_mul	-0,0603635236620903	0,0110593726858497
    """

    def __init__(self):
        super(F_res18_35_cifar10_4_relu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.gelu(F.relu(x))) * F.relu(x)


class F_res18_35_cifar10_0_gelu(nn.Module):
    """
    5
    95,936	leaky_relu-max0-sig_mul-gelu-gelu-sig_mul	-0,0114085273817182	0,0457472577691078
    """

    def __init__(self):
        super(F_res18_35_cifar10_0_gelu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.gelu(F.sigmoid(F.leaky_relu(x)) * F.relu(x))) * F.gelu(x)


class F_res18_35_cifar10_0_silu(nn.Module):
    """
    5
    96,624	gelu-gelu-sig_mul-gelu-gelu-sig_mul
    """

    def __init__(self):
        super(F_res18_35_cifar10_0_silu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.gelu(F.sigmoid(F.gelu(x)) * F.gelu(x))) * F.gelu(x)


class F_res18_35_cifar10_1_gelu(nn.Module):
    """
    5
    51	0,012121730195712	82,9322295955208	96,576	gelu-gelu-beta_mix-gelu-gelu-sig_mul	-0,0465634316205978
    """

    def __init__(self):
        super(F_res18_35_cifar10_1_gelu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.gelu(F.gelu(x))) * F.gelu(x)


class F_res18_35_cifar10_2_silu(nn.Module):
    """
    5
    47	0,012923679398488	88,4576020326931	96,304	gelu-gelu-min-gelu-gelu-sig_mul

    """

    def __init__(self):
        super(F_res18_35_cifar10_2_silu, self).__init__()

    def forward(self, x):
        return F.gelu(F.gelu(x))


class F_res18_35_cifar10_4_silu(nn.Module):
    """
    5
    47	0,012923679398488	88,4576020326931	96,304	gelu-gelu-min-gelu-gelu-sig_mul

    """

    def __init__(self):
        super(F_res18_35_cifar10_4_silu, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.gelu(0.4925 * F.leaky_relu(x) + 0.5075 * F.gelu(x))) * F.gelu(x)


class F_gpt2_openweb_gelu_2_5(nn.Module):
    """
    seed 2, layer 5
    """

    def __init__(self):
        super(F_gpt2_openweb_gelu_2_5, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.leaky_relu(F.gelu(x))) * F.gelu(x)


class F_rand_1(nn.Module):
    """
    seed 2, layer 5
    tanh-max0-beta_mix-sigmoid-x_sq-mul
    """

    def __init__(self):
        super(F_rand_1, self).__init__()

    def forward(self, x):
        return torch.sigmoid(0.5 * torch.tanh(x) + 0.5 * torch.relu(x)) * torch.pow(x, 2).clamp(min=-10, max=10)


class F_rand_2(nn.Module):
    """
    seed 2, layer 5
    max0-neg_x-exp_sub_sq-x_sq-elu-mul
    # todo rerun
    """

    def __init__(self):
        super(F_rand_2, self).__init__()

    def forward(self, x):
        return torch.pow(torch.exp(-torch.pow(torch.relu(x) + x, 2)), 2).clamp(min=-10, max=10) * F.elu(x)


class F_rand_3(nn.Module):
    """
    seed 2, layer 5
    max0-exp-add-beta_mul-beta_mul-sig_mul
    """

    def __init__(self):
        super(F_rand_3, self).__init__()

    def forward(self, x):
        return torch.sigmoid(torch.relu(x) + torch.exp(x).clamp(min=-10, max=10)) * x


class F_rand_4(nn.Module):
    """
    seed 2, layer 5
    max0-beta-add-neg_x-leaky_relu-right
    todo rerun
    """

    def __init__(self):
        super(F_rand_4, self).__init__()

    def forward(self, x):
        return F.leaky_relu(x)


class F_rand_5(nn.Module):
    """
    seed 2, layer 5
    asinh-x_sq-sig_mul-max0-beta-beta_mix
    """

    def __init__(self):
        super(F_rand_5, self).__init__()

    def forward(self, x):
        return 0.5 * torch.relu(torch.sigmoid(torch.asinh(x)) * torch.pow(x, 2)) + 0.5


class F_rand_6(nn.Module):
    """
    seed 2, layer 5
    tanh-silu-left-neg_x-id-min
    """

    def __init__(self):
        super(F_rand_6, self).__init__()

    def forward(self, x):
        return torch.minimum(-torch.tanh(x), x)


class F_rand_7(nn.Module):
    """
    seed 2, layer 5
    id-exp-min-gelu-silu-max
    todo rerun
    """

    def __init__(self):
        super(F_rand_7, self).__init__()

    def forward(self, x):
        return torch.maximum(F.gelu(torch.minimum(x, torch.exp(x).clamp(-10, 10))), F.silu(x))


class F_rand_11(nn.Module):
    """
    seed 2, layer 5
    max0-tanh-max-x_sq-beta-beta_mix
    """

    def __init__(self):
        super(F_rand_11, self).__init__()

    def forward(self, x):
        return 0.5 * torch.pow(torch.maximum(torch.relu(x), torch.tanh(x)), 2).clamp(min=-10, max=10) + 0.5


class F_rand_10(nn.Module):
    """
    seed 2, layer 5
    beta_add-silu-sig_mul-elu-neg_x-substrat
    todo rerun
    """

    def __init__(self):
        super(F_rand_10, self).__init__()

    def forward(self, x):
        return F.elu(torch.sigmoid(x + 1) * F.silu(x)) + x


class F_rand_8(nn.Module):
    """
    seed 2, layer 5
    asinh-beta_add-exp_sub_abs-max0-tanh-sig_mul
    """

    def __init__(self):
        super(F_rand_8, self).__init__()

    def forward(self, x):
        return torch.sigmoid(torch.relu(torch.exp(-torch.pow(torch.asinh(x) - (x + 1), 2)))) * torch.tanh(x)


class F_rand_9(nn.Module):
    """
    seed 2, layer 5
    id-id-min-elu-id-right
    """

    def __init__(self):
        super(F_rand_9, self).__init__()

    def forward(self, x):
        return x


class F_rand_12(nn.Module):
    """
    seed 2, layer 5
    beta_add-neg_x-min-x_sq-max0-mul
    """

    def __init__(self):
        super(F_rand_12, self).__init__()

    def forward(self, x):
        return torch.pow(torch.min(x + 1, - x), 2).clamp(-10, 10) * torch.relu(x)


class F_rand_13(nn.Module):
    """
    seed 2, layer 5
    x_sq-beta_add-beta_mix-tanh-gelu-beta_mix
    todo rerun
    """

    def __init__(self):
        super(F_rand_13, self).__init__()

    def forward(self, x):
        return 0.5 * torch.tanh(0.5 * torch.pow(x, 2).clamp(-10, 10) + 0.5 * (x + 1)) + 0.5 * F.gelu(x)


class F_rand_14(nn.Module):
    """
    seed 2, layer 5
    exp-asinh-right-tanh-gelu-left
    """

    def __init__(self):
        super(F_rand_14, self).__init__()

    def forward(self, x):
        return torch.asinh(torch.tanh(x))


class F_rand_15(nn.Module):
    """
    seed 2, layer 5
    tanh-silu-max-max0-beta_mul-mul
    todo rerun
    """

    def __init__(self):
        super(F_rand_15, self).__init__()

    def forward(self, x):
        return torch.relu(torch.maximum(torch.tanh(x), F.silu(x))) * x


class F_rand_16(nn.Module):
    """
    seed 2, layer 5
    sigmoid-gelu-exp_sub_sq-x_cub-silu-max
    todo rerun
    """

    def __init__(self):
        super(F_rand_16, self).__init__()

    def forward(self, x):
        return torch.maximum(torch.pow(torch.exp(-torch.pow(torch.sigmoid(x) - F.gelu(x), 2)), 3).clamp(-10, 10),
                             F.silu(x))


class F_rand_17(nn.Module):
    """
    seed 2, layer 5
    silu-x_sq-add-sigmoid-beta_mul-beta_mix
    todo rerun
    """

    def __init__(self):
        super(F_rand_17, self).__init__()

    def forward(self, x):
        return 0.5 * torch.sigmoid(F.silu(x) + torch.square(x).clamp(-10, 10)) + 0.5 * x


class F_rand_18(nn.Module):
    """
    seed 2, layer 5
    tanh-tanh-left-silu-id-max
    todo reru
    """

    def __init__(self):
        super(F_rand_18, self).__init__()

    def forward(self, x):
        return torch.maximum(F.silu(torch.tanh(x)), x)


class F_rand_19(nn.Module):
    """
    seed 2, layer 5
    x_sq-silu-exp_sub_sq-x_sq-sigmoid-mul
    todo rerun
    """

    def __init__(self):
        super(F_rand_19, self).__init__()

    def forward(self, x):
        return torch.pow(torch.exp(-torch.pow(torch.pow(x, 2) - F.silu(x), 2)), 2).clamp(-10, 10) * torch.sigmoid(x)


class F_rand_20(nn.Module):
    """
    seed 2, layer 5
    gelu-neg_x-right-leaky_relu-elu-left
    """

    def __init__(self):
        super(F_rand_20, self).__init__()

    def forward(self, x):
        return F.leaky_relu(-x)


class F_rand_21(nn.Module):
    """
    seed 2, layer 5
    gelu-neg_x-right-leaky_relu-elu-left
    """

    def __init__(self):
        super(F_rand_21, self).__init__()

    def forward(self, x):
        return F.leaky_relu(-x)


class F_rand_22(nn.Module):
    """
    seed 2, layer 5
    id-x_cub-right-leaky_relu-max0-substrat
    """

    def __init__(self):
        super(F_rand_22, self).__init__()

    def forward(self, x):
        return F.leaky_relu(torch.pow(x, x).clamp(-10, 10)) - torch.relu(x)


class F_rand_23(nn.Module):
    """
    seed 2, layer 5
    beta-asinh-mul-sigmoid-tanh-mul
    """

    def __init__(self):
        super(F_rand_23, self).__init__()

    def forward(self, x):
        return torch.sigmoid(torch.asinh(x)) * torch.tanh(x)


class F_rand_24(nn.Module):
    """
    seed 2, layer 5
    exp-neg_x-beta_mix-x_sq-tanh-beta_mix
    """

    def __init__(self):
        super(F_rand_24, self).__init__()

    def forward(self, x):
        return 0.5 * torch.pow(0.5 * torch.exp(x).clamp(-10, 10) - 0.5 * x, 2).clamp(-10, 10) + 0.5 * torch.tanh(x)


class F_rand_25(nn.Module):
    """
    seed 2, layer 5
    max0-tanh-min-sigmoid-asinh-left
    """

    def __init__(self):
        super(F_rand_25, self).__init__()

    def forward(self, x):
        return torch.sigmoid(torch.minimum(torch.relu(x), torch.tanh(x)))


class F_rand_26(nn.Module):
    """
    seed 2, layer 5
    x_cub-beta_mul-right-id-elu-beta_mix
    todo rerun
    """

    def __init__(self):
        super(F_rand_26, self).__init__()

    def forward(self, x):
        return 0.5 * F.elu(x) + 0.5 * x


class F_rand_27(nn.Module):
    """
    seed 2, layer 5
    x_sq-asinh-right-leaky_relu-neg_x-left
    """

    def __init__(self):
        super(F_rand_27, self).__init__()

    def forward(self, x):
        return F.leaky_relu(torch.asinh(x))


class F_rand_28(nn.Module):
    """
    seed 2, layer 5
    id-exp-exp_sub_sq-neg_x-neg_x-add
    """

    def __init__(self):
        super(F_rand_28, self).__init__()

    def forward(self, x):
        return -torch.exp(-torch.pow(x - torch.exp(x).clamp(-10, 10), 2)) - x


class F_rand_29(nn.Module):
    """
    seed 2, layer 5
    silu-asinh-add-exp-x_cub-left
    todo rerun
    """

    def __init__(self):
        super(F_rand_29, self).__init__()

    def forward(self, x):
        return torch.exp(F.silu(x) + torch.asinh(x)).clamp(-10, 10)


class F_rand_30(nn.Module):
    """
    seed 2, layer 5
    beta_add-tanh-exp_sub_abs-gelu-x_sq-substrat
    todo rerun
    """

    def __init__(self):
        super(F_rand_30, self).__init__()

    def forward(self, x):
        return F.gelu(torch.exp(-torch.abs(x + 1 - torch.tanh(x)))) - torch.square(x).clamp(-10, 10)


class F_rand_31(nn.Module):
    """
    seed 2, layer 5
    exp-beta_mul-add-beta-silu-mul
    todo rerun
    """

    def __init__(self):
        super(F_rand_31, self).__init__()

    def forward(self, x):
        return F.silu(x)


class F_rand_32(nn.Module):
    """
    seed 2, layer 5
    exp-x_sq-beta_mix-beta-leaky_relu-left
    todo rerun
    """

    def __init__(self):
        super(F_rand_32, self).__init__()

    def forward(self, x):
        return torch.ones_like(x)


class F_rand_33(nn.Module):
    """
    seed 2, layer 5
    silu-leaky_relu-beta_mix-gelu-tanh-beta_mix
    todo rerun
    """

    def __init__(self):
        super(F_rand_33, self).__init__()

    def forward(self, x):
        return 0.5 * F.gelu(0.5 * F.silu(x) + 0.5 * F.leaky_relu(x)) + 0.5 * torch.tanh(x)


class F_rand_34(nn.Module):
    """
    seed 2, layer 5
    beta_mul-elu-substrat-elu-tanh-max
    todo rerun
    """

    def __init__(self):
        super(F_rand_34, self).__init__()

    def forward(self, x):
        return torch.maximum(F.elu(x - F.elu(x)), torch.tanh(x))


class F_rand_35(nn.Module):
    """
    seed 2, layer 5
    gelu-asinh-beta_mix-max0-leaky_relu-left
    todo rerun
    """

    def __init__(self):
        super(F_rand_35, self).__init__()

    def forward(self, x):
        return F.relu(0.5 * F.gelu(x) + 0.5 * torch.asinh(x))


class F_vit_tiny_35_cifar10_0_gelu_False_True(nn.Module):
    """
    single False, reg True
    silu-silu-sig_mul-gelu-silu-min
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_0_gelu_False_True, self).__init__()

    def forward(self, x):
        return torch.minimum(F.gelu(F.sigmoid(F.silu(x)) * F.silu(x)), F.silu(x))


class F_vit_tiny_35_cifar10_1_gelu_False_True(nn.Module):
    """
    single False, reg True
    x_sq-silu-beta_mix-silu-silu-beta_mix,-0.23659631609916687,-0.23433831334114075
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_1_gelu_False_True, self).__init__()

    def forward(self, x):
        return 0.4417 * F.silu(0.4411 * torch.square(x).clamp(max=10, min=-10) + 0.5589 * F.silu(x)) + 0.5583 * F.silu(x)


class F_vit_tiny_35_cifar10_2_gelu_False_True(nn.Module):
    """
    single False, reg True
    silu-silu-sig_mul-gelu-silu-sig_mul
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_2_gelu_False_True, self).__init__()

    def forward(self, x):
        return F.sigmoid(F.gelu(F.sigmoid(F.silu(x)) * F.silu(x))) * F.silu(x)


class F_vit_tiny_35_cifar10_0_gelu_True_False(nn.Module):
    """
    single True, reg False
    silu-silu-mul-gelu-silu-beta_mix,0.013736099004745483,-0.1489002853631973
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_0_gelu_True_False, self).__init__()

    def forward(self, x):
        return 0.4628 * F.gelu(F.silu(x).clamp(max=10) * F.silu(x).clamp(max=10)) + 0.5372 * F.silu(x)


class F_vit_tiny_35_cifar10_1_gelu_True_False(nn.Module):
    """
    single True, reg False
    x_sq-silu-beta_mix-silu-silu-beta_mix,0.02587084472179413,-0.20284701883792877,
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_1_gelu_True_False, self).__init__()

    def forward(self, x):
        return 0.4495 * F.silu(0.5065 * torch.square(x).clamp(max=10) + 0.4935 * F.silu(x)) + 0.5505 * F.silu(x)


class F_vit_tiny_35_cifar10_2_gelu_True_False(nn.Module):
    """
    single True, reg False
    x_sq-silu-beta_mix-silu-silu-beta_mix,0.02587084472179413,-0.20284701883792877,
    x_sq-silu-beta_mix-silu-silu-beta_mix,-0.15883474051952362,-0.1581798791885376
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_2_gelu_True_False, self).__init__()

    def forward(self, x):
        return 0.4605 * F.silu(0.4604 * torch.square(x).clamp(max=10) + 0.5396 * F.silu(x)) + 0.5395 * F.silu(x)


class F_vit_tiny_35_cifar10_3_gelu_True_False(nn.Module):
    """
    single True, reg False
    (7)
    beta_mul-silu-sig_mul-x_sq-silu-beta_mix,-0.00380430999211967,-0.04204856976866722
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_3_gelu_True_False, self).__init__()

    def forward(self, x):
        return 0.4895 * torch.square(F.sigmoid(0.5192 * x) * F.silu(x)).clamp(max=10) + 0.5105 * F.silu(x)


class F_vit_tiny_35_cifar10_4_gelu_True_False(nn.Module):
    """
    single True, reg False
    (7)
    beta_mul-beta_mul-sig_mul-x_sq-silu-sig_mul
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_4_gelu_True_False, self).__init__()

    def forward(self, x):
        return torch.sigmoid(torch.square(F.sigmoid(0.5848 * x) * 0.6962 * x)) * F.silu(x)


class F_vit_tiny_35_cifar10_5_gelu_True_False(nn.Module):
    """
    clamped
    single True, reg False
    (7)
    beta_mul-beta_mul-sig_mul-x_sq-silu-sig_mul
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_5_gelu_True_False, self).__init__()

    def forward(self, x):
        return torch.sigmoid(torch.square(F.sigmoid(0.5848 * x) * 0.6962 * x).clamp(max=10)) * F.silu(x)



class F_vit_tiny_35_cifar10_0_gelu_True_True(nn.Module):
    """
    clamped
    single True, reg True
    silu-silu-mul-gelu-silu-beta_mix,0.043144434690475464,-0.10630371421575546
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_0_gelu_True_True, self).__init__()

    def forward(self, x):
        return 0.4734 * F.gelu(F.silu(x).clamp(max=10) * F.silu(x).clamp(max=10)) + 0.5266 * F.silu(x)


class F_vit_tiny_35_cifar10_1_gelu_True_True(nn.Module):
    """
    clamped
    single True, reg True
    x_sq-silu-beta_mix-silu-silu-beta_mix,0.016691723838448524,-0.13864928483963013
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_1_gelu_True_True, self).__init__()

    def forward(self, x):
        return 0.4654 * F.silu(0.5042 * torch.square(x).clamp(max=10) + 0.4958 * F.silu(x)) + 0.5346 * F.silu(x)


class F_vit_tiny_35_cifar10_2_gelu_True_True(nn.Module):
    """
    clamped
    single True, reg True
    silu-silu-sig_mul-x_sq-silu-beta_mix,-0.006191887892782688,-0.07105468958616257
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_2_gelu_True_True, self).__init__()

    def forward(self, x):
        return 0.4822 * torch.square(F.sigmoid(F.silu(x)) * F.silu(x)).clamp(max=10) + 0.5178 * F.silu(x)


class F_vit_tiny_35_cifar10_3_gelu_True_True(nn.Module):
    """
    clamped
    single True, reg True
    beta_mul-silu-sig_mul-x_sq-silu-beta_mix,-0.002418921794742346,-0.07281883805990219
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_3_gelu_True_True, self).__init__()

    def forward(self, x):
        return 0.4818 * torch.square(F.sigmoid(0.5221 * x) * F.silu(x)).clamp(max=10) + 0.5182 * F.silu(x)


class F_vit_tiny_35_cifar10_4_gelu_True_True(nn.Module):
    """
    single True, reg True
    silu-silu-sig_mul-x_sq-silu-sig_mul
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_4_gelu_True_True, self).__init__()

    def forward(self, x):
        return F.sigmoid(torch.square(F.sigmoid(F.silu(x)) * F.silu(x))) * F.silu(x)


class F_vit_tiny_35_cifar10_4_gelu_True_True(nn.Module):
    """
    clamped
    single True, reg True
    silu-silu-sig_mul-x_sq-silu-sig_mul
    """

    def __init__(self):
        super(F_vit_tiny_35_cifar10_4_gelu_True_True, self).__init__()

    def forward(self, x):
        return F.sigmoid(torch.square(F.sigmoid(F.silu(x)) * F.silu(x)).clamp(max=10)) * F.silu(x)