from ground_truth import lamb_oseen_velocity, lamb_oseen_vortex


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math


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


def train_lamb_oseen(config, net_u):
    # build and train
    device = config.device
    net_u.to(device)
    Loss = torch.zeros(config.num_iterations + 1)
    optimizer = optim.Adam(net_u.parameters(), config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    r = torch.Tensor([[0.0, 1.0], [-1.0, 0.0]]).to(device)
    delta_t = (config.total_time - 0.0) / config.num_time_interval
    # begin optimization iteration
    for step in range(config.num_iterations + 1):
        net_u.train()
        xy = 2.0 * config.truncation * torch.rand((config.batch_size_xy, 2)).to(device) - config.truncation
        nu = (config.nu_max - config.nu_min) * torch.rand((config.batch_size_nu, 1)).to(device) + config.nu_min
        sqrt_nu = torch.sqrt(nu)
        xy_nu = torch.concat([xy.repeat(config.batch_size_nu, 1), torch.log10(nu).repeat(1, config.batch_size_xy).reshape(-1, 1)], dim=1)
        loss = torch.Tensor([0.0]).to(device)
        sum_k = torch.zeros(size=(config.batch_size_nu, config.num_sample, 2)).to(device)
        for k in range(0, config.num_time_interval):
            u_t = net_u[k](xy_nu)
            if k == 0:
                sum_k = sqrt_nu.reshape(-1, 1, 1) * math.sqrt(2 * delta_t) * torch.randn(1, config.num_sample, 2).to(device)
            else:
                sum_k_nu = torch.concat([sum_k.reshape(-1, 2), torch.log10(nu).repeat(config.num_sample, 1)], dim=1)
                sum_k += net_u[k-1](sum_k_nu).detach().reshape(sum_k.shape) * delta_t + sqrt_nu.reshape(-1, 1, 1) * \
                    math.sqrt(2 * delta_t) * torch.randn(1, config.num_sample, 2).to(device)
            temp = (xy.reshape(1, config.batch_size_xy, 1, 2) - sum_k.reshape(config.batch_size_nu, 1, config.num_sample, 2)).reshape(-1, 2)
            u_hat = 1/(2 * math.pi) * (torch.mm(temp, r)/((temp**2).sum(axis=1).view(-1, 1) + EPS)).\
                view(config.batch_size_nu * config.batch_size_xy, config.num_sample, 2).mean(axis=1)
            loss += torch.sort(((u_t - u_hat) ** 2).sum(axis=1), descending=True)[0][5:].mean()
        Loss[step] = loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 20 == 0:
            net_u.eval()
            xy_grid = torch.Tensor([(i / config.lattice, j / config.lattice) for i in range(-config.lattice * config.truncation, config.truncation * config.lattice + 1)
                                    for j in range(-config.lattice * config.truncation, config.truncation * config.lattice + 1)]).to(device)
            sol = torch.zeros(config.num_time_interval, xy_grid.shape[0], 2)
            for t in range(1, config.num_time_interval + 1):
                sol[t - 1, :, :] = lamb_oseen_velocity(xy_grid, t * delta_t, config.nu)
            xy_grid_nu = torch.concat([xy_grid, math.log10(config.nu) * torch.ones(xy_grid.shape[0], 1).to(device)], dim=1)
            u_pre = net_u[-1](xy_grid_nu)
            u_real = sol[-1, :, :]
            print('###########################', step, loss.detach().item(), (torch.linalg.norm((u_pre.cpu() - u_real)) / torch.linalg.norm(u_real)).item())
    return net_u, Loss