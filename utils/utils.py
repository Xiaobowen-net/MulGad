import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from torch.autograd import Variable
import numpy as np
def save_checkpoint(model, path,file,checkpoint):
    print("Saving model ...")
    torch.save(model.state_dict(), os.path.join(path,str(os.path.splitext(file)[0]) + checkpoint))
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


