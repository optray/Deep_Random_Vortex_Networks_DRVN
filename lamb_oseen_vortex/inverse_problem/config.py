import torch


class Config(object):
    # params in network
    device = torch.device('cuda:0')
    lr = 0.01
    batch_size = 100
    step_size = 500
    gamma = 0.2
    num_iterations = 2000

    # params in 2d-NSE
    truncation = 2
    lattice = 64
    nu = 0.5
    total_time = 1
    num_time_interval = 40


def get_config(name):
    try:
        return globals()[name + 'Config']
    except KeyError:
        raise KeyError("Config for the required problem not found.")
