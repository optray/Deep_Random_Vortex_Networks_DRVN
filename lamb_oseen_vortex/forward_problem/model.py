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
    xy_grid = torch.Tensor([(i / config.lattice, j / config.lattice) for i in range(-config.lattice * config.truncation, config.truncation * config.lattice + 1)
                            for j in range(-config.lattice * config.truncation, config.truncation * config.lattice + 1)]).to(config.device)
    # build and train
    device = config.device
    net_u.to(device)
    optimizer = optim.Adam(net_u.parameters(), config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    r = torch.Tensor([[0.0, 1.0], [-1.0, 0.0]]).to(device)
    delta_t = (config.total_time - 0.0) / config.num_time_interval
    training_loss_record = torch.zeros(config.num_iterations//10 + 1)
    u_pre_record = torch.zeros(config.num_iterations//10 + 1, config.num_time_interval)
    u_real = [lamb_oseen_velocity(xy_grid.cpu(), t * delta_t, config.nu) for t in range(1, config.num_time_interval + 1)]
    # begin optimization iteration
    for step in range(config.num_iterations+1):
        net_u.train()
        xy = 2.0 * config.truncation * torch.rand((config.batch_size, 2)).to(device) - config.truncation
        loss = torch.Tensor([0.0]).to(device)
        sum_k = torch.zeros(size=(config.num_sample, 2)).to(device)
        for k in range(0, config.num_time_interval):
            u_t = net_u[k](xy)
            sum_k += (k != 0) * net_u[k-1](sum_k).detach() * delta_t + math.sqrt(2 * config.nu * delta_t) * \
                         torch.randn(config.num_sample, 2).to(device)
            temp = xy.repeat(1, config.num_sample).view(-1, 2) - sum_k.repeat(config.batch_size, 1)
            u_hat = 1/(2 * math.pi) * (torch.mm(temp, r)/((temp**2).sum(axis=1).view(-1, 1) + EPS)).\
                view(config.batch_size, config.num_sample, 2).mean(axis=1)
            loss += torch.sort(((u_t - u_hat) ** 2).sum(axis=1), descending=True)[0][5:].mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0:
            training_loss_record[step//10] = loss.detach().cpu()
            for k in range(0, config.num_time_interval):
                u_pre = net_u[k](xy_grid).detach().cpu()
                u_pre_record[step//10, k] = torch.linalg.norm((u_pre - u_real[k])) / torch.linalg.norm(u_real[k])
            print('###########################', step, training_loss_record[step//10], u_pre_record[step//10, :].mean())
    return net_u, training_loss_record, u_pre_record