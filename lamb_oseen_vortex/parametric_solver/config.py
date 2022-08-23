import torch


class Config(object):
    # params in network
    device = torch.device('cuda:0')
    lr = 0.001
    batch_size_nu = 500
    batch_size_xy = 40
    step_size = 500
    gamma = 0.5
    num_iterations = 10000
    num_hiddens_u = [3] + [512] * 6 + [2]
    activate_u = torch.nn.ReLU(inplace=True)
    num_sample = 200

    # params in 2d-NSE
    truncation = 2
    lattice = 64
    nu = 0.01
    nu_min = 0.001
    nu_max = 0.6
    total_time = 1
    num_time_interval = 40


def get_config(name):
    try:
        return globals()[name + 'Config']
    except KeyError:
        raise KeyError("Config for the required problem not found.")
