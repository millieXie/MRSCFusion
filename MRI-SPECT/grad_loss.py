import torch
import numpy as np
import cv2
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import linalg as LA

# class GradientLoss(nn.Module):
#
#     def __init__(self,input_ch,output_ch):
#         super(GradientLoss, self).__init__()
#
#         sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).cuda()
#         sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).cuda()
#         sobel_3x = torch.Tensor(1, input_ch, 3, 3).cuda()
#         sobel_3y = torch.Tensor(1, input_ch, 3, 3).cuda()
#         sobel_3x[:, 0:input_ch, :, :] = sobel_x
#         sobel_3y[:, 0:input_ch, :, :] = sobel_y
#         self.conv_hx = nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv_hy = nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv_hx.weight = torch.nn.Parameter(sobel_3x)
#         self.conv_hy.weight = torch.nn.Parameter(sobel_3y)
#
#
#     def forward(self, X, Y):  #X:fuse,Y=source
#         b, c, w, h = X.shape
#         X = X.cuda()
#         Y = Y.cuda()
#         X_hx = self.conv_hx(X)
#         X_hy = self.conv_hy(Y)
#         G_X = torch.abs(X_hx) + torch.abs(X_hy)
#         # compute gradient of Y
#         Y_hx = self.conv_hx(Y)
#         self.conv_hx.train(False)
#         Y_hy = self.conv_hy(Y)
#         self.conv_hy.train(False)
#         G_Y = torch.abs(Y_hx) + torch.abs(Y_hy)
#         # loss = F.mse_loss(G_X, G_Y, size_average=True)
#         Fs = torch.norm(G_X-G_Y, p='fro', dim=None, keepdim=False, out=None, dtype=None)
#         Fs = torch.pow(Fs, 2)/(w*h*b*c)
#         # for i in range(b):
#         #     for j in range(c):
#         #         torch.norm()
#         #         Fs= LA.norm(torch.norm((G_Y[i][j].cpu().numpy())-(G_X[i][j].cpu().numpy())), -1, ksize=3, ord='fro')
#         #         Fs = torch.pow(2,Fs)
#         return Fs

class GradientLoss(nn.Module):
    def __init__(self,input_ch,output_ch):
        super(GradientLoss, self).__init__()
        Laplacian = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).cuda()
        Laplacian_3C = torch.Tensor(1, input_ch, 3, 3).cuda()
        Laplacian_3C[:, 0:input_ch, :, :] = Laplacian
        self.conv_la = nn.Conv2d(input_ch,output_ch, kernel_size=3, stride=1, padding=1, bias=False).cuda()
        self.conv_la.weight = torch.nn.Parameter(Laplacian_3C)
    def forward(self, X, Y):
        b, c, w, h = X.shape
        X = X.cuda()
        Y = Y.cuda()
        X_la = self.conv_la(X)
        Y_la = self.conv_la(Y)
        # compute gradient of Y
        self.conv_la.train(False)
        Fs = torch.norm(X_la - Y_la, p='fro', dim=None, keepdim=False, out=None, dtype=None)
        Fs = torch.pow(Fs, 2) / (w * h * b * c)
        return Fs

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class Gradloss(nn.Module):
    def __init__(self):
        super(Gradloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        # image_y=image_vis[:,:1,:,:]
        vi_grad=self.sobelconv(image_vis)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(vi_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_grad=100*loss_grad
        return loss_grad


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss