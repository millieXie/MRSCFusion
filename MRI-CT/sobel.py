import numpy.random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor


def convloss(x, channels=3):
    x = Variable(torch.from_numpy(x.astype(np.float32))).permute(2, 0, 1)
    kernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    weight = nn.Parameter(data=kernel, requires_grad=False)
    x = F.conv2d(x.unsqueeze(1), weight=weight, padding=1)
    x = x.squeeze(1).permute(1, 2, 0).numpy()
    return 1-x.mean()

