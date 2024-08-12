import torch
import torch.nn.functional as F
import itertools

import numpy as np
import random 


from torch.nn.utils import clip_grad_norm_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MonotonicityLoss(torch.nn.Module):
    def __init__(self):
        super(MonotonicityLoss, self).__init__()

    def forward(self, x, y):
        # Ensure that outputs2 is always greater than outputs1
        # add sign to not interpret the magnitude of the slope
        return torch.mean(torch.relu(y - x))


class DiscountedHistoryLoss(torch.nn.Module):
    def __init__(self, discount_factor=0.9, max_history=2):
        super(DiscountedHistoryLoss, self).__init__()
        self.discount_factor = discount_factor
        self.max_history = max_history
        self.history = [F.relu(torch.linspace(-10, 10, 100).to(device))]

    def forward(self, ft_x):
        # Calculate loss based on historical values
        loss = 0.0
        for i, historical_ft_x in enumerate(self.history[::-1]):
            # Apply discounting
            discount = self.discount_factor ** (i + 1)
            # Calculate MSE loss and add to the total loss
            mse_loss = F.mse_loss(ft_x, historical_ft_x, reduction='mean')
            loss += discount * mse_loss

        # Update history
        if len(self.history) >= self.max_history:
            self.history.pop(0)  # Remove oldest entry if max history reached
        # self.history.append(ft_x.detach())  # Detach from current computation graph

        return loss

    def add_func(self, ft_x):
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(ft_x.detach())  # Detach from current computation graph


class Architecture:
    def __init__(self, model, ac_func, valid_loader, criterion, epochs, scaler, lr=0.01, arch_weight_decay=0.001):

        self.epochs = epochs
        self.scaler = scaler

        self.ac_func = ac_func
        self.lr = lr

        self.valid_cache = valid_loader
        self.valid_loader = iter(valid_loader)

        self.criterion = criterion
        self.criterion_2 = torch.nn.MSELoss(reduce="mean")
        self.criterion_monotonicity = MonotonicityLoss()

        self.optimizer_arch = torch.optim.Adam(self.ac_func.get_search_parameters(), lr=self.lr, betas=(0.5, 0.999),
                                               weight_decay=arch_weight_decay)
    

        # self.scheduler_arch = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_arch, T_max=self.epochs-1, eta_min=self.lr/10)

        self.sym_weight = 1e-2
        self.disc_weight = 6.125 * 1e-2

        self.histor_loss = DiscountedHistoryLoss(0.9, 10)

    def step(self, model, item, grad_acumm=1, batch_idx=0, loader_len=1):

        img_arch, label_arch = item
        img_arch, label_arch = img_arch.to(device), label_arch.to(device, non_blocking=True)

        # with torch.cuda.amp.autocast(enabled=True):
        pred_arch = model(img_arch)
        loss_arch = self.criterion(pred_arch, label_arch)
        
        loss = loss_arch / grad_acumm

        # loss.backward()
        self.scaler.scale(loss).backward()
        # clip_grad_norm_(self.ac_func.arch_parameters(), 5.0)
        if batch_idx % grad_acumm == (grad_acumm - 1) or batch_idx == (loader_len - 1):
            # self.optimizer_arch.step()
            self.scaler.step(self.optimizer_arch)
            self.scaler.update()
            self.optimizer_arch.zero_grad()

    # todo not reg but loss
    def sym_reg(self):
        # x-symmetry
        x_sym = 1 / torch.sum(self.ac_func(torch.linspace(-10, 0, 1000).to(device)) + self.ac_func(
            torch.linspace(0, 10, 1000).to(device)), dim=0)
        return x_sym

    def disc_reg(self):
        d_disc_curr = self.ac_func(torch.linspace(-10, 10, 1000).to(device)) - self.ac_func.forward_discretized(
            torch.linspace(-10, 10, 1000).to(device))
        return torch.abs(d_disc_curr).mean()

    def progressive_shrink(self):
        raise NotImplementedError
    
    def random_seed(self, seed=1337, rank=0):
        np.random.seed(seed + rank)
        random.seed(seed + rank)
        torch.manual_seed(seed + rank)
 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed + rank)
