# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from reinmax import reinmax



def wrapper_class(original_class):
    class WrapperClass(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.activ = nn.ReLU()

            self.num_unary = len(self.activ.unary[0])
            self.num_binary = len(self.activ.binary[0])
        
            self.nargmx = torch.zeros_like(self.alphas_normal.argmax(dim=-1))
            self.rargmx = torch.zeros_like(self.alphas_reduce.argmax(dim=-1))

            self.prun_interval = 5
            self.no_change_epochs = 0

            self.usparsity = 0.5
            self.bsparsity = 0.5

            self.stop_search = False

            self.warm_starting = True

            self._initialize_alphas()

            #### reg
            self.reg_type = 'l2'
            self.reg_scale = 1e-3


        def change_activation(self, new_activation=None):

            def replace_layers(model, layer_old, layer_new):
                for name, module in model.named_children():
                    if len(list(module.children())) > 0:
                        replace_layers(module, layer_old, layer_new)
                        
                    if isinstance(module, layer_old):
                        setattr(model, name, layer_new)

            if new_activation is not None:
                self.activ = new_activation
                replace_layers(self, nn.GELU, new_activation)
                replace_layers(self, nn.ReLU, new_activation)
            else:
                replace_layers(self, nn.GELU, self.activ)
                replace_layers(self, nn.ReLU, self.activ)

        def sample_reinmax(self, weights):
            p_hard, p_soft = reinmax(weights, 1)
            return p_hard

        
        def forward(self, x):
            
            self.activ.fwd_ops = [self.activ.ops[i][self.activ.mask[i]] for i in range(len(self.activ.mask))]
            self.activ.fwd_alphas = [self.sample_reinmax(self.activ.alpha[i][self.activ.mask[i]]) for i in range(len(self.activ.mask))]            

            x = super().forward(x)
            return x

    return WrapperClass

