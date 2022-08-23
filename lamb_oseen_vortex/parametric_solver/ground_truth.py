import torch
import math


EPS = 1e-10


def lamb_oseen_velocity(x, t, nu):
    xi = x / math.sqrt(nu * t)
    r = torch.Tensor([[0.0, 1.0], [-1.0, 0.0]]).to(x.device)
    return torch.mm(xi, r) * (1.0 - torch.exp(- 0.25 * (xi ** 2).sum(axis=1)).view(-1, 1)) \
           / (2 * torch.pi * math.sqrt(nu * t) * (xi ** 2).sum(axis=1) + EPS).view(-1, 1)


def lamb_oseen_vortex(x, t, nu):
    xi = x / math.sqrt(nu * t)
    return torch.exp(- 0.25 * (xi ** 2).sum(axis=1)).reshape(-1, 1) / (4 * math.pi * nu * t)