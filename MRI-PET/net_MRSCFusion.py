import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FusionNet import RGBD
from network_swinir import RSTB
import math

EPSILON = 1e-10


def var(x, dim=0):  # 方差
    x_zero_meaned = x - x.mean(dim).expand_as(x)  # dim=0按行求均值，结果是一个列向量，通过复制扩展成一个和x相同形状
    return x_zero_meaned.pow(2).mean(dim)  # 均方差


class MultConst(nn.Module):
    def forward(self, input):
        return 255 * input


class UpsampleReshape_eval(torch.nn.Module):  # 上采样加重新变形
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)  # 输出是输入的两个

    def forward(self, x1, x2):  #
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0  # 图像居中
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right % 2 is 0.0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot % 2 is 0.0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2

class PaddedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, ks, stride):
        super().__init__()
        # Custom Padding Calculation
        if isinstance(ks, tuple):  # 判断ks是否为tuple类型
            k_h, k_w = ks  # k_h=(ks)[0],k_w=(ks)[1]
        else:
            k_h = ks
            k_w = ks
        if isinstance(stride, tuple):
            s_h, s_w = stride
        else:
            s_h = stride
            s_w = stride
        pad_h, pad_w = k_h - s_h, k_w - s_w
        pad_up, pad_left = pad_h // 2, pad_w // 2
        pad_down, pad_right = pad_h - pad_up, pad_w - pad_left
        self.pad = nn.ZeroPad2d([pad_left, pad_right, pad_up, pad_down])  # 对tensor用0进行边界填充
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=ks, stride=stride, bias=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x



# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out

# Dense Block unit
# light version
class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        # out_channels_def = 16
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []
        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


class ConvLayer(torch.nn.Module):  # 卷积层
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


# Convolution operation
class f_ConvLayer(torch.nn.Module):  # 融合卷积层
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(f_ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        # out = self.batch_norm(out)
        out = F.relu(out, inplace=True)
        return out


class FusionBlock_res(torch.nn.Module):
    def __init__(self, channels, img_size, index):
        super(FusionBlock_res, self).__init__()

        self.conv_first_pet = nn.Conv2d(channels,channels, 3, 1, 1)
        self.conv_first_mri = nn.Conv2d(channels, channels, 3, 1, 1)
        self.axial_fusion = nn.Sequential(f_ConvLayer(2 * channels, channels, 1, 1))
        self.conv_fusion_pet = nn.Sequential(f_ConvLayer(channels, channels, 3, 1))
        self.conv_fusion_mri = nn.Sequential(f_ConvLayer(channels, channels, 3, 1))

        block = []  # 空间分支
        block += [f_ConvLayer(2 * channels, channels, 1, 1),
                  f_ConvLayer(channels, channels, 3, 1),
                  f_ConvLayer(channels, channels, 3, 1)]
        self.bottelblock = nn.Sequential(*block)
        self.image_size = img_size

        self.swin_IR1_pet = RSTB(dim=channels, input_resolution=(img_size, img_size), depth=3, num_heads=4,
                            window_size=8, img_size=img_size)
        self.swin_IR1_mri = RSTB(dim=channels, input_resolution=(img_size, img_size), depth=3, num_heads=4,
                             window_size=8, img_size=img_size)
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.rgbd_ir = RGBD(channels, channels)
        self.rgbd_vi = RGBD(channels, channels)
        self.conv_after_body = nn.Conv2d(2*channels, channels, 3, 1, 1)


    def forward(self, x_ir, x_vi):

        x_vi_t = self.conv_first_mri(x_vi)
        x_ir_t = self.conv_first_pet(x_ir)

        x_ir_t = torch.flatten(x_ir_t, 2).permute(0, 2, 1)
        x_vi_t = torch.flatten(x_vi_t, 2).permute(0, 2, 1)

        x_ir_t = self.swin_IR1_pet(x_ir_t,(self.image_size,self.image_size))
        x_vi_t = self.swin_IR1_mri(x_vi_t,(self.image_size,self.image_size))

        x_ir_t = x_ir_t.permute(0, 2, 1).view(-1, x_ir.shape[-3], x_ir.shape[-2], x_ir.shape[-1])
        x_vi_t = x_vi_t.permute(0, 2, 1).view(-1, x_vi.shape[-3], x_vi.shape[-2], x_vi.shape[-1])


        x_vi_t += x_vi
        x_ir_t += x_ir
        a_init = torch.cat([x_vi_t, x_ir_t], 1)
        a_init = self.conv_after_body(a_init)

        #
        x_cvi = self.conv_fusion_mri(x_vi)
        x_cir = self.conv_fusion_pet(x_ir)
        x_cvi = self.rgbd_ir(x_cvi)
        x_cir = self.rgbd_vi(x_cir)
        out = torch.cat([x_cvi, x_cir], 1)
        out = self.bottelblock(out)
        out = a_init + out

        return out


# Fusion network, 4 groups of features
class Fusion_network(nn.Module):  # 融合网络
    def __init__(self, nC, fs_type):
        super(Fusion_network, self).__init__()
        self.fs_type = fs_type
        img_size = [256, 128, 64, 32]

        self.fusion_block1 = FusionBlock_res(nC[0], img_size[0], 0)
        self.fusion_block2 = FusionBlock_res(nC[1], img_size[1], 1)
        self.fusion_block3 = FusionBlock_res(nC[2], img_size[2], 2)
        self.fusion_block4 = FusionBlock_res(nC[3], img_size[3], 3)

    def forward(self, en_ir, en_vi):
        f1_0 = self.fusion_block1(en_ir[0], en_vi[0])
        f2_0 = self.fusion_block2(en_ir[1], en_vi[1])
        f3_0 = self.fusion_block3(en_ir[2], en_vi[2])
        f4_0 = self.fusion_block4(en_ir[3], en_vi[3])

        return [f1_0, f2_0, f3_0, f4_0]



# NestFuse network - light, no desnse
class NestFuse_light2_nodense(nn.Module):
    def __init__(self, nb_filter, input_nc=3, output_nc=3, deepsupervision=True):
        super(NestFuse_light2_nodense, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        # encoder
        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)

        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)

        # short connection
        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)

        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            # self.conv4 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)

    def encoder(self, input):
        x = self.conv0(input)
        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0(self.pool(x1_0))
        x3_0 = self.DB3_0(self.pool(x2_0))
        x4_0 = self.DB4_0(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]

    def decoder_train(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))
        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]

    def decoder_eval(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]


class RFN_decoder(nn.Module):  # 解码器
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
        super(RFN_decoder, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)

        # short connection
        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)

        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            # self.conv4 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)

    def decoder_train(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))
        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]

    def decoder_eval(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]

