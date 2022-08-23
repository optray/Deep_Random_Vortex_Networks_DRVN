import torch
import math


EPS = 1e-10


def lamb_oseen_vortex(x, t, nu):
    xi = x / math.sqrt(nu * t)
    r = torch.Tensor([[0.0, 1.0], [-1.0, 0.0]]).to(x.device)
    return torch.mm(xi, r) * (1.0 - torch.exp(- 0.25 * (xi ** 2).sum(axis=1)).view(-1, 1)) \
           / (2 * torch.pi * math.sqrt(nu * t) * (xi ** 2).sum(axis=1) + EPS).view(-1, 1)


def lamb_oseen_omega(x, t, nu):
    xi = x / math.sqrt(nu * t)
    return torch.exp(- 0.25 * (xi ** 2).sum(axis=1)).reshape(-1, 1) / (4 * math.pi * nu * t)


def u_gth(w, xy, lattice, truncation):
    batch_size = xy.shape[0]
    device = w.device
    r = torch.Tensor([[0.0, 1.0], [-1.0, 0.0]]).to(device)
    xy_grid = torch.Tensor([(i / lattice, j / lattice) for i in range(-lattice * truncation, truncation * lattice + 1)
                            for j in range(-lattice * truncation, truncation * lattice + 1)]).to(device)
    temp = xy.repeat(1, xy_grid.shape[0]).view(-1, 2) - xy_grid.repeat(batch_size, 1)
    return 1 / (2 * math.pi) * (torch.mm(temp, r)/((temp**2).sum(axis=1).view(-1, 1) + EPS)
                                * w.repeat(batch_size, 2)).view(batch_size, -1, 2).mean(axis=1)