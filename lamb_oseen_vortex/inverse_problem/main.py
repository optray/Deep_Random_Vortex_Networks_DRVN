from config import get_config
from ground_truth import lamb_oseen_vortex, u_gth
from model import FNN, Para, train_inverse

import os
import torch
import matplotlib.pyplot as plt

plt.rcParams["animation.html"] = "jshtml"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    cfg = get_config('')
    device = cfg.device
    xy_grid = torch.Tensor([(i / cfg.lattice, j / cfg.lattice) for i in range(-cfg.lattice * cfg.truncation, cfg.truncation * cfg.lattice + 1)
                            for j in range(-cfg.lattice * cfg.truncation, cfg.truncation * cfg.lattice + 1)]).to(device)
    w0 = torch.zeros(xy_grid.shape[0], 1).to(device)
    w0[xy_grid.shape[0] // 2, 0] = 1.0 * xy_grid.shape[0]
    delta_t = cfg.total_time / cfg.num_time_interval

    for nu_test in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
        cfg.nu = nu_test
        for i in range(5):
            net_u = torch.load('net/net_u_'+ str(0), map_location=device)
            para_nu = Para()
            nu, nu_list = train_inverse(cfg, w0, net_u, para_nu)
            torch.save(nu_list, 'nu_list_nu_'+str(cfg.nu) + '_' + str(i))
            print(nu, cfg.nu)