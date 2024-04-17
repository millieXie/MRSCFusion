import cv2
import torch.nn.functional
from torch import linalg as LA


def FM(input):
    """

    :param input: shape batch c w h
    :return: Feature measurement
    """
    input1 = input.clone().detach()
    b, c, w, h = input1.size()
    G = torch.tensor([0], dtype=torch.float64)
    # 计算单个batch的Feature measurement
    for i in range(b):
        for j in range(c):
            G[0] += pow(LA.norm(torch.from_numpy(cv2.Laplacian(input1[i][j].cpu().numpy(), -1, ksize=3)), ord='fro'),2)
    return G/(b*c*w*h)


def c_softmax(G1, G2, c):
    return torch.nn.functional.softmax(torch.cat((G1/c, G2/c)))


def weight_block(input1 ,input2, c):
    """

    :param input1: tensor infrared image (经过初步的提取)
    :param input2: tensor visible image
    :param c: constant c
    :return: 损失函数的权重
    """
    G1 = FM(input1)
    G2 = FM(input2)
    w = c_softmax(G1, G2, c)
    return w



