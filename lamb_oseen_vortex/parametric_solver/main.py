from config import get_config
from ground_truth import lamb_oseen_velocity, lamb_oseen_vortex
from model import FNN, train_lamb_oseen

import os
import torch
import math
import matplotlib.pyplot as plt

plt.rcParams["animation.html"] = "jshtml"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    cfg = get_config('')
    device = cfg.device
    for i in range(5):
        net_u = torch.nn.ModuleList([FNN(cfg, cfg.activate_u, cfg.num_hiddens_u) for _ in range(cfg.num_time_interval)])
        net_u, Loss = train_lamb_oseen(cfg, net_u)
        torch.save(net_u, 'net_u_'+str(i))
        torch.save(Loss, 'Loss_' + str(i))

    xy_grid = torch.Tensor([(i / cfg.lattice, j / cfg.lattice) for i in range(-cfg.lattice * cfg.truncation, cfg.truncation * cfg.lattice + 1)
                            for j in range(-cfg.lattice * cfg.truncation, cfg.truncation * cfg.lattice + 1)]).to(device)
    delta_t = cfg.total_time / cfg.num_time_interval
    sol = torch.zeros(cfg.num_time_interval, xy_grid.shape[0], 2)
    for t in range(1, cfg.num_time_interval+1):
        sol[t - 1, :, :] = lamb_oseen_velocity(xy_grid, t * delta_t, cfg.nu)
    xy_grid_nu = torch.concat([xy_grid, math.log10(cfg.nu) * torch.ones(xy_grid.shape[0], 1).to(device)], dim=1)
    u_pre = net_u[-1](xy_grid_nu)
    u_real = sol[-1, :, :]
    print(torch.linalg.norm((u_pre.cpu() - u_real)) / torch.linalg.norm(u_real))