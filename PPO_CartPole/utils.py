import numpy as np, random, torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def moving_average(x, w=10):
    return np.convolve(x, np.ones(w)/w, mode='valid')
