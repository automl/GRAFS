import copy
import copyreg
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch.nn.functional as F

from reinmax import reinmax


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Identitiy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Sign(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -x


class Abs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


class Pow2(nn.Module):
    def __init__(self, inf=10.0):
        super().__init__()
        self.inf = torch.tensor(inf)

    def forward(self, x):
        return torch.pow(torch.clamp(x, max=torch.sqrt(self.inf.to(x.device)), min=-torch.sqrt(self.inf.to(x.device))), 2)


class Pow3(nn.Module):
    def __init__(self, inf=10.0):
        super().__init__()
        self.inf = torch.tensor(inf)

    def forward(self, x):
        return torch.pow(torch.clamp(x, max=torch.pow(self.inf.to(x.device), 1/3), min=-torch.pow(self.inf.to(x.device), 1/3)), 3)
        

class Sqrt(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = torch.tensor(1e-3)

    def forward(self, x):
        x = torch.sqrt(torch.clamp(x, min=self.eps.to(x.device)))
        return x


class BetaMul(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.normal(torch.tensor(1.), torch.tensor(.25)).to(device), requires_grad=True)

    def forward(self, x):
        return x * self.beta


class BetaAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.normal(torch.tensor(1.), torch.tensor(.25)).to(device), requires_grad=True)

    def forward(self, x):
        return x + self.beta


class Beta(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.normal(torch.tensor(1.), torch.tensor(.25)).to(device), requires_grad=True)

    def forward(self, x):
        return self.beta * torch.ones_like(x)


class LogAbsEps(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = torch.tensor(eps)

    def forward(self, x):
        return torch.log(torch.abs(x) + self.eps)
        

class Exp(nn.Module):
    def __init__(self, inf=10.0):
        super().__init__()
        self.inf = torch.tensor(inf)

    def forward(self, x):
        return torch.exp(torch.clamp(x, max=torch.log(self.inf.to(x.device))))
        

class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class Cos(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cos(x)


class Sinh(nn.Module):
    def __init__(self, inf=10.0):
        super().__init__()
        self.inf = torch.tensor(inf)

    def forward(self, x):
        return torch.sinh(torch.clamp(x, max=torch.asinh(self.inf.to(x.device)), min=-torch.asinh(self.inf.to(x.device))))
        

class Cosh(nn.Module):
    def __init__(self, inf=10.0):
        super().__init__()
        self.inf = torch.tensor(inf)

    def forward(self, x):
        return torch.cosh(torch.clamp(x, max=torch.acosh(self.inf.to(x.device)), min=-torch.acosh(self.inf.to(x.device))))


class Tanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)


class Asinh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)


class Atan(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.atan(x)


class Sinc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sinc(x)


class Max0(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)


class Min0(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -torch.relu(-x)


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)


class LogExp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log1p(torch.exp(x))


class Exp2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.exp(-torch.pow(x, 2))
        return z


class Erf(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.erf(x)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.functional.F.silu(x)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.functional.F.gelu(x)

class ELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.functional.F.elu(x)

    
class LeakyRELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.functional.F.leaky_relu(x)




class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2


class Mul(nn.Module):
    def __init__(self, inf=10.0):
        super().__init__()
        self.inf = torch.tensor(inf)

    def forward(self, x1, x2):
        return torch.clamp(x1, max=torch.sqrt(self.inf.to(x1.device)), min=-torch.sqrt(self.inf.to(x1.device))) * torch.clamp(x2, max=torch.sqrt(self.inf.to(x2.device)), min=-torch.sqrt(self.inf.to(x2.device)))


class Sub(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 - x2


class Div(nn.Module):
    def __init__(self, inf=10.0):
        super().__init__()
        self.inf = torch.tensor(inf)

    def forward(self, x1, x2):
        z = torch.nan_to_num(x1 / x2, posinf=self.inf, neginf=-self.inf)
        return z


class Maximum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.max(x1, x2)


class Minimum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.min(x1, x2)


class SigMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.sigmoid(x1) * x2


class AbsExp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.exp(-torch.abs(x1 - x2))


class Pow2Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.exp(-torch.pow(x1 - x2, 2))


class BetaMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)

    def forward(self, x1, x2):
        x = self.beta.sigmoid() * x1 + (1 - self.beta.sigmoid()) * x2
        return x
    
class Left(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1


class Right(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x2




class RobustFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Perform your forward operation here
        result = input  # example operation
        ctx.save_for_backward(input)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        # Compute the gradient
        grad_input = grad_output.clone()  # example gradient computation

        # Replace NaN and Inf values in grad_input with zero
        grad_input[torch.isnan(grad_input)] = 0.0
        grad_input[torch.isinf(grad_input)] = torch.sign(grad_input[torch.isinf(grad_input)]) * 100.

        return grad_input


class ActivationCell(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_type = "drnas"
        self.robust_f = RobustFunction()

        self.eval_genotype = None
        self.best_genotype = None

        self.unaries = [np.array([  
            Identitiy(), 
            Sign(), 
            Abs(),
            Pow2(), 
            Pow3(), 
            Sqrt(),
            BetaAdd(), 
            BetaMul(), 
            Beta(),
            #   LogAbsEps(), 
            Exp(), 
            #   Sin(), 
            #   Cos(), 
            Sinh(),
            #   Cosh(), 
            Tanh(), 
            Asinh(), 
            Atan(),
            #   Sinc(), 
            Max0(), 
            Min0(),  
            Sigmoid(), 
            LogExp(),
            #   Exp2(), 
            Erf(),
            SiLU(), 
            GELU(), 
            ELU(), 
            LeakyRELU()
            ]) for _ in range(4)]

            

        self.binaries = [np.array([
                Add(), 
                Sub(), 
                Mul(), 
                Maximum(), 
                Minimum(), 
                SigMul(), 
                BetaMix(),
                Left(),
                Right()
                ]) for _ in range(2)]


        self.learnable_parameter = [op for u in self.unaries for oplist in u for op in list(set(oplist.parameters()))] + [op for b in self.binaries for oplist in b for op in list(set(oplist.parameters()))]

        self.clamp_value = 10


        self.ops = [self.unaries[0], self.unaries[1], self.binaries[0], self.unaries[2], self.unaries[3], self.binaries[1]]


        self.mask = [
            np.ones(len(self.ops[0])).astype(bool),
            np.ones(len(self.ops[1])).astype(bool),
            np.ones(len(self.ops[2])).astype(bool),
            np.ones(len(self.ops[3])).astype(bool),
            np.ones(len(self.ops[4])).astype(bool),
            np.ones(len(self.ops[5])).astype(bool),
        ]

        self.mask_dummy = copy.deepcopy(self.mask)


        self.alpha = [
            nn.Parameter(torch.rand(len(self.ops[0])).to(device) * 1e-6, requires_grad=True),
            nn.Parameter(torch.rand(len(self.ops[1])).to(device) * 1e-6, requires_grad=True),
            nn.Parameter(torch.rand(len(self.ops[2])).to(device) * 1e-6, requires_grad=True),
            nn.Parameter(torch.rand(len(self.ops[3])).to(device) * 1e-6, requires_grad=True),
            nn.Parameter(torch.rand(len(self.ops[4])).to(device) * 1e-6, requires_grad=True),
            nn.Parameter(torch.rand(len(self.ops[5])).to(device) * 1e-6, requires_grad=True),
        ]


    def forward(self, x):
        if self.forward_type == "drnas":
            return self.forward_drnas(x)
        elif self.forward_type == "reinmax":
            return self.forward_reinmax(x)
        elif self.forward_type == "discretized":
            return self.forward_discretized(x)
        elif self.forward_type == "softmax":
            return self.forward_softmax(x)
        elif self.forward_type == "eval":
            return self.forward_eval(x)
        elif self.forward_type == "relu":
            return F.relu(x)
        elif self.forward_type == "gelu":
            return F.gelu(x)
        elif self.forward_type == "elu":
            return F.elu(x)
        elif self.forward_type == "silu":
            return F.silu(x)
        elif self.forward_type == "leakyrelu":
            return F.leaky_relu(x)
        else:
            raise KeyError

    def forward_drnas(self, x):
        u_0 = sum(op(x) * w_i for op, w_i in
                  zip(self.ops[0][self.mask[0]],
                      self.sample_alphas(self.alpha[0][self.mask[0]].to(device))))
        u_1 = sum(op(x) * w_i for op, w_i in
                  zip(self.ops[1][self.mask[1]],
                      self.sample_alphas(self.alpha[1][self.mask[1]].to(device))))
        b_0 = sum(op(u_0, u_1) * w_i for op, w_i in
                  zip(self.ops[2][self.mask[2]],
                      self.sample_alphas(self.alpha[2][self.mask[2]].to(device))))

        u_2 = sum(op(b_0) * w_i for op, w_i in
                  zip(self.ops[3][self.mask[3]],
                      self.sample_alphas(self.alpha[3][self.mask[3]].to(device))))
        u_3 = sum(op(x) * w_i for op, w_i in
                  zip(self.ops[4][self.mask[4]],
                      self.sample_alphas(self.alpha[4][self.mask[4]].to(device))))
        b_1 = sum(op(u_2, u_3) * w_i for op, w_i in
                  zip(self.ops[5][self.mask[5]],
                      self.sample_alphas(self.alpha[5][self.mask[5]].to(device))))
        return b_1
    
    def forward_reinmax(self, x):
        u_0 = sum(op(x) * w_i for op, w_i in
                  zip(self.ops[0][self.mask[0]],
                      self.sample_reinmax(self.alpha[0][self.mask[0]].to(device))))
        u_1 = sum(op(x) * w_i for op, w_i in
                  zip(self.ops[1][self.mask[1]],
                      self.sample_reinmax(self.alpha[1][self.mask[1]].to(device))))
        b_0 = sum(op(u_0, u_1) * w_i for op, w_i in
                  zip(self.ops[2][self.mask[2]],
                      self.sample_reinmax(self.alpha[2][self.mask[2]].to(device))))

        u_2 = sum(op(b_0) * w_i for op, w_i in
                  zip(self.ops[3][self.mask[3]],
                      self.sample_reinmax(self.alpha[3][self.mask[3]].to(device))))
        u_3 = sum(op(x) * w_i for op, w_i in
                  zip(self.ops[4][self.mask[4]],
                      self.sample_reinmax(self.alpha[4][self.mask[4]].to(device))))
        b_1 = sum(op(u_2, u_3) * w_i for op, w_i in
                  zip(self.ops[5][self.mask[5]],
                      self.sample_reinmax(self.alpha[5][self.mask[5]].to(device))))
        return b_1
    
    def forward_softmax(self, x):
        u_0 = sum(op(x) * w_i for op, w_i in
                  zip(self.ops[0][self.mask[0]],
                      self.apply_softmax(self.alpha[0][self.mask[0]].to(device))))
        u_1 = sum(op(x) * w_i for op, w_i in
                  zip(self.ops[1][self.mask[1]],
                      self.apply_softmax(self.alpha[1][self.mask[1]].to(device))))
        b_0 = sum(op(u_0, u_1) * w_i for op, w_i in
                  zip(self.ops[2][self.mask[2]],
                      self.apply_softmax(self.alpha[2][self.mask[2]].to(device))))

        u_2 = sum(op(b_0) * w_i for op, w_i in
                  zip(self.ops[3][self.mask[3]],
                      self.apply_softmax(self.alpha[3][self.mask[3]].to(device))))
        u_3 = sum(op(x) * w_i for op, w_i in
                  zip(self.ops[4][self.mask[4]],
                      self.apply_softmax(self.alpha[4][self.mask[4]].to(device))))
        b_1 = sum(op(u_2, u_3) * w_i for op, w_i in
                  zip(self.ops[5][self.mask[5]],
                      self.apply_softmax(self.alpha[5][self.mask[5]].to(device))))
        return b_1

    # def forward_drnas(self, x):
    #     u_0 = sum(op(x) * w for op, w in zip(self.fwd_ops[0], self.fwd_alphas[0]))
    #     u_1 = sum(op(x) * w for op, w in zip(self.fwd_ops[1], self.fwd_alphas[1]))
    #     b_0 = sum(op(u_0, u_1) * w for op, w in zip(self.fwd_ops[2], self.fwd_alphas[2]))

    #     u_2 = sum(op(b_0) * w for op, w in zip(self.fwd_ops[3], self.fwd_alphas[3]))
    #     u_3 = sum(op(x) * w for op, w in zip(self.fwd_ops[4], self.fwd_alphas[4]))
    #     b_1 = sum(op(u_2, u_3) * w for op, w in zip(self.fwd_ops[5], self.fwd_alphas[5]))
    #     return b_1

    def forward_discretized(self, x):
        arg_max_0 = torch.argmax(torch.masked_select(self.alpha[0], torch.tensor(self.mask[0]).to(device)))
        u_0 = self.ops[0][self.mask[0]][arg_max_0](x)
        arg_max_1 = torch.argmax(torch.masked_select(self.alpha[1], torch.tensor(self.mask[1]).to(device)))
        u_1 = self.ops[1][self.mask[1]][arg_max_1](x)
        arg_max_b_1 = torch.argmax(torch.masked_select(self.alpha[2], torch.tensor(self.mask[2]).to(device)))
        b_0 = self.ops[2][self.mask[2]][arg_max_b_1](u_0, u_1)

        arg_max_2 = torch.argmax(torch.masked_select(self.alpha[3], torch.tensor(self.mask[3]).to(device)))
        u_2 = self.ops[3][self.mask[3]][arg_max_2](b_0)
        arg_max_3 = torch.argmax(torch.masked_select(self.alpha[4], torch.tensor(self.mask[4]).to(device)))
        u_3 = self.ops[4][self.mask[4]][arg_max_3](x)
        arg_max_b_2 = torch.argmax(torch.masked_select(self.alpha[5], torch.tensor(self.mask[5]).to(device)))
        b_1 = self.ops[5][self.mask[5]][arg_max_b_2](u_2, u_3)
        # print(arg_max_0, arg_max_1, arg_max_b_1, arg_max_2, arg_max_3, arg_max_b_2)
        return b_1
    
    def forward_eval(self, x):

        core1 = self.eval_genotype[:3]
        core2 = self.eval_genotype[3:]

        temp1 = core1[0](x)
        temp2 = core1[1](x)
        temp3 = core1[2](temp1, temp2)

        temp4 = core2[0](temp3)
        temp5 = core2[1](x) 
        temp6 = core2[2](temp4, temp5)

        return temp6

    def sample_alphas(self, weights):
        beta = F.elu(weights) + 1
        weights = torch.distributions.dirichlet.Dirichlet(beta).rsample()
        return weights
    
    def sample_reinmax(self, weights):
            p_hard, p_soft = reinmax(weights, 1)
            return p_hard
    
    def apply_softmax(self, weights):
        weights = F.softmax(weights)
        return weights

    def delete_ops(self):
        del_flag = [sum(self.mask[idx]) for idx in range(len(self.alpha))]
        a_max = max(del_flag)
        del_flag = [i >= a_max for i in del_flag]

        for idx, alpha_i in enumerate(self.alpha):
            if sum(self.mask[idx]) > 1 and del_flag[idx]:
                threshold = torch.inf
                del_index = 0
                for i, alpha_i_k in enumerate(alpha_i):
                    if alpha_i_k < threshold and self.mask[idx][i]:
                        threshold = alpha_i_k
                        del_index = i
                self.mask[idx][del_index] = False
                break
                
        print(self.mask)

    def drop_op(self, d):

            alphas = [a.clone() for a in self.alpha]

            for _ in range(d):
            
                for i in range(len(self.mask)):
                    alphas[i][~self.mask[i]] = torch.inf
                del_flag = [sum(m) for m in self.mask]
                index_edge = del_flag.index(max(del_flag))
                index_op = alphas[index_edge].argmin().item()
                self.mask[index_edge][index_op] = False

            
            print('mask lens =', [sum(m) for m in self.mask])
    
    def drop_op_edgewise(self, nu, nb):
        '''drop ops and _arch_parameters() 
        but then have to do soemthing for the genotype! 
        which hasn't been done yet.'''

        alphas = [a.clone() for a in self.alpha]


        for i in [0,1,3,4]:
            for _ in range(nu):
                alphas[i][~self.mask[i]] = torch.inf
                index_op = alphas[i].argmin(dim=-1)
                self.mask[i][index_op] = False


        for i in [2,5]:
            for _ in range(nb):
                alphas[i][~self.mask[i]] = torch.inf
                index_op = alphas[i].argmin(dim=-1)
                self.mask[i][index_op] = False

        print('mask lens =', [sum(m) for m in self.mask])


    def set_alpha(self, alpha):
        self.alpha = alpha

    def plot(self, name_extension=str(time.time()), width=10):
        linestyle_tuple = {
            'loosely dotted': (0, (1, 10)),
            'dotted': (0, (1, 1)),
            'densely dotted': (0, (1, 1)),
            'long dash with offset': (5, (10, 3)),
            'loosely dashed': (0, (5, 10)),
            'dashed': (0, (5, 5)),
            'densely dashed': (0, (5, 1)),

            'loosely dashdotted': (0, (3, 10, 1, 10)),
            'dashdotted': (0, (3, 5, 1, 5)),
            'densely dashdotted': (0, (3, 1, 1, 1)),

            'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
            'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
            'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
        }

        # Getting a list of all named colors in matplotlib
        all_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

        # Filtering out the green colors
        greens = {name: color for name, color in all_colors.items() if 'green' in name.lower()}

        # Selecting 6 different shades of green
        selected_greens = list(greens.items())[:6]

        self.mask_dummy = copy.deepcopy(self.mask)
        x = torch.linspace(-width, width, 1000)

        with torch.no_grad():
            plt.plot(x.detach().numpy(), torch.relu(x.to(device)).cpu().detach().numpy(),
                     label="relu", alpha=0.45, linestyle=linestyle_tuple['long dash with offset'],
                     color=selected_greens[0][1])
            plt.plot(x.detach().numpy(), F.elu(x.to(device)).cpu().detach().numpy(),
                     label="elu", alpha=0.45, linestyle=linestyle_tuple['densely dashdotted'],
                     color=selected_greens[1][1])
            plt.plot(x.detach().numpy(), F.silu(x.to(device)).cpu().detach().numpy(),
                     label="silu", alpha=0.45, linestyle=linestyle_tuple['loosely dashdotdotted'],
                     color=selected_greens[2][1])
            plt.plot(x.detach().numpy(), F.gelu(x.to(device)).cpu().detach().numpy(),
                     label="gelu", alpha=0.45, linestyle=linestyle_tuple['loosely dashdotted'],
                     color=selected_greens[3][1])
            plt.plot(x.detach().numpy(), torch.sigmoid(x.to(device)).cpu().detach().numpy(),
                     label="sigmoid", alpha=0.45, linestyle=linestyle_tuple['dashed'], color=selected_greens[4][1])
            plt.plot(x.detach().numpy(), self(x.to(device)).cpu().detach().numpy(),
                     label="function", c="r")
            plt.plot(x.detach().numpy(), self.forward_discretized(x.to(device)).cpu().detach().numpy(),
                     label="disc_function", c="b")
        plt.legend()
        # todo: idea bound disc to non disc function as reg
        plt.savefig(f"plots/plot_{name_extension}.png")
        plt.close()
        self.mask = copy.deepcopy(self.mask_dummy)

    # def parameters(self, recurse: bool = False):
    #     for param in self.learnable_parameter.parameters():
    #         if param is not None:
    #             yield param

    def arch_parameters(self, recurse: bool = False):
        for param in self.alpha:
            if param is not None:
                yield param

    def get_search_parameters(self, recurse: bool = False):
        for param in self.alpha + self.learnable_parameter:
            if param is not None:
                yield param


    def __str__(self):
        res = ""
        for i, mask_i in enumerate(self.mask):
            for k, flag in enumerate(mask_i):
                if flag:
                    res += f"{self.ops_names[i][k]} "
            res += "\n\n"
        return res

    # def disc_func(self):
    #     res = ""
    #     for i, mask_i in enumerate(self.mask):
    #         max_idx = None
    #         max_value = -np.inf
    #         for k, flag in enumerate(mask_i):
    #             if flag and max_value < self.alpha[i][k]:
    #                 max_value = self.alpha[i][k]
    #                 max_idx = k
    #         res += f"-{self.ops_names[i][max_idx]}"
    #     return res[1:]
    
    def genotype(self):
        gene = []
        for i, mask_i in enumerate(self.mask):
            max_idx = None
            max_value = -np.inf
            for k, flag in enumerate(mask_i):
                if flag and max_value < self.alpha[i][k]:
                    max_value = self.alpha[i][k]
                    max_idx = k
            gene.append(self.ops[i][max_idx])
        return gene

    def betas_str(self):
        with torch.no_grad():
            content = ','.join(f"{float(p.cpu())}" for u in self.ops for op in u for p in op.parameters())
        return content

    def __len__(self):
        return sum([sum(mask_i) for mask_i in self.mask])
