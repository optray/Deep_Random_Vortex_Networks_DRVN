from ground_truth import u_gth, lamb_oseen_vortex

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.tensorboard import SummaryWriter

EPS = 1e-10


class Dense(nn.Module):
    def __init__(self, d_in, d_out, activate):
        super(Dense, self).__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.activate = activate
        nn.init.normal_(self.linear.weight, std=1.0 / np.sqrt(d_in + d_out))

    def forward(self, x):
        x = self.linear(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class FNN(nn.Module):
    def __init__(self, config, activate, num_hiddens):
        super(FNN, self).__init__()
        self._config = config
        self.layers = [Dense(num_hiddens[i - 1], num_hiddens[i], activate=activate) for i in
                       range(1, len(num_hiddens) - 1)]
        self.layers += [Dense(num_hiddens[-2], num_hiddens[-1], activate=None)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class Para(nn.Module):
    def __init__(self):
        super(Para, self).__init__()
        # self.nu = torch.nn.Parameter(torch.Tensor([0.5]))
        self.nu = torch.nn.Parameter(torch.abs(torch.randn(1)))

    def forward(self):
        return self.nu


def train_inverse(config, w0, net_u, para_nu):
    writer = SummaryWriter('logs')
    # build and train
    device = config.device
    para_nu.to(device)
    optimizer = optim.Adam(para_nu.parameters(), config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    delta_t = (config.total_time - 0.0) / config.num_time_interval
    log_nu = para_nu()
    nu_list = []
    # begin optimization iteration
    xy = 2.0 * config.truncation * torch.rand((config.batch_size, 2)).to(device) - config.truncation
    u_xy = [lamb_oseen_vortex(xy, (k + 1) * delta_t, config.nu) for k in range(0, config.num_time_interval)]
    for step in range(config.num_iterations + 1):
        xy_nu = torch.concat((xy, log_nu * torch.ones(config.batch_size, 1).to(device)), dim=1)
        nu = 10 ** log_nu
        loss = torch.Tensor([0.0]).to(device)
        # sum_k = torch.zeros(size=(config.num_sample, 2)).to(device)
        for k in range(0, config.num_time_interval):
            u_t = net_u[k](xy_nu)
            loss += ((u_t[::1, :] - u_xy[k][::1, :]) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        nu_list.append(10 ** log_nu.item())
        loss_nu = torch.abs((10 ** log_nu - config.nu) / config.nu)

        writer.add_scalar("train loss", loss.item(), step)
        writer.add_scalar("nu loss", loss_nu.item(), step)

        if step % 20 == 0:
            print('###########################', step, loss.item(), nu.data.item(), loss_nu.item())
    return nu.item(), nu_list