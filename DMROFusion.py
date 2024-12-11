import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from args import img_size
import kornia

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        )
    def __call__(self, x):
        out = self.body(x)
        return out + x

class Conv_layer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(Conv_layer, self).__init__()
        self.initial_0 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=True),
            nn.PReLU(),
        )
    def forward(self, x):
        x = self.initial_0(x)
        return x

class ConcatConv(nn.Module):
    def __init__(self, channel, reduction=1):
        super(ConcatConv, self).__init__()
        # spatial attention
        self.channel = channel
        self.initial_0 = nn.Sequential(
            nn.Conv2d(2 * channel, channel, 3, padding=1, bias=True),
            nn.PReLU())
    def forward(self, x1, x2):
        x12 = torch.cat([x1, x2], 1)
        x12 = self.initial_0(x12)
        return x12

class UpDown_Module(nn.Module):
    def __init__(self, in_channels):
        super(UpDown_Module, self).__init__()
        self.in_channels = in_channels
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.intra_L = Conv_layer(self.in_channels)
        self.intra_H = Conv_layer(self.in_channels)
        self.inter_L = ConcatConv(self.in_channels)
        self.inter_H = ConcatConv(self.in_channels)

    def forward(self, X_l, X_h):
        X_ha, X_la = self.intra_H(X_h), self.intra_L(X_l)
        X_h2la, X_l2ha = self.h2g_pool(X_ha), self.up(X_la)
        X_l_out = self.inter_L(X_la, X_h2la)
        X_h_out = self.inter_H(X_ha, X_l2ha)
        return X_l_out, X_h_out

class UpDown_Module0(nn.Module):
    def __init__(self, in_channels):
        super(UpDown_Module0, self).__init__()
        self.h2g_convdow = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=True),
            nn.PReLU(),
        )
        self.h2g_convdow1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=True),
            nn.PReLU(),
        )
        self.in_channels = in_channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.intra_L = Conv_layer(self.in_channels)
        self.intra_H = Conv_layer(self.in_channels)

        self.inter_L = ConcatConv(self.in_channels)
        self.inter_H = ConcatConv(self.in_channels)

    def forward(self, x):
        X_h, X_l = x, self.h2g_convdow(x)
        X_ha, X_la = self.intra_H(X_h), self.intra_L(X_l)
        X_h2la, X_l2ha = self.h2g_convdow1(X_ha), self.up(X_la)
        X_l_out = self.inter_L(X_la, X_h2la)
        X_h_out = self.inter_H(X_ha, X_l2ha)
        return X_l_out, X_h_out

class ConvLSTM(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel):

        super().__init__()
        pad_x = 1
        self.conv_xf = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)
        self.conv_xi = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)
        self.conv_xo = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)
        self.conv_xj = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)

        pad_h = 1
        self.conv_hf = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hi = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_ho = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hj = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

    def forward(self, x, h, c):

        if h is None and c is None:
            i = F.sigmoid(self.conv_xi(x))
            o = F.sigmoid(self.conv_xo(x))
            j = F.tanh(self.conv_xj(x))
            c = i * j
            h = o * c
        else:
            f = F.sigmoid(self.conv_xf(x) + self.conv_hf(h))
            i = F.sigmoid(self.conv_xi(x) + self.conv_hi(h))
            o = F.sigmoid(self.conv_xo(x) + self.conv_ho(h))
            j = F.tanh(self.conv_xj(x) + self.conv_hj(h))
            c = f * c + i * j
            h = o * F.tanh(c)

        return h, h, c

class GCM(nn.Module):
    def __init__(self):
        super(GCM, self).__init__()
        self.Blur = kornia.filters.BoxBlur((11, 11))

        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, dilation=1, groups=1, bias=True)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0, dilation=1, groups=1, bias=True)
        self.conv4 = torch.nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv5 = nn.Conv2d(32, 16, 1, bias=False)
        self.conv6a = nn.Conv2d(16, 1, 1, bias=False)
        self.conv6b = nn.Conv2d(16, 1, 1, bias=False)
        self.conv6c = nn.Conv2d(16, 1, 1, bias=False)
        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self,img):
        img_blur = self.Blur(img)

        img_res = img-img_blur
        f1 = self.conv1(img)
        f1 = self.prelu(f1)
        f2 = self.conv2(f1)
        f2 = self.prelu(f2)
        f3 = self.conv3(f2)
        f3 = self.prelu(f3)
        f4 = self.conv4(f3)

        g1 = self.avg_pool(f4)
        g2 = self.max_pool(f4)

        g12 = self.conv5(torch.cat([g1, g2], 1))
        g12 = self.prelu(g12)

        W1 = self.conv6a(g12)
        W2 = self.conv6c(g12)
        b = self.conv6b(g12)

        W1 = self.tanh(W1)
        W2 = self.tanh(W2)
        b = self.tanh(b)

        y_output = W1*img_blur + W2*img_res+b
        return y_output
        
class Encoder_HYP(nn.Module):
    def __init__(self):
        super(Encoder_HYP, self).__init__()
        r = 16

        # Here, 1*1 convolution layer is used to replace fully-connected layer
        self.convp1 = nn.Conv2d(1, 16, 1, bias=False)
        self.convp2 = nn.Conv2d(16, 16, 1, bias=False)
        self.convp3 = nn.Conv2d(16, 16, 1, bias=False)

        self.convp4a = nn.Conv2d(16, 16, 1, bias=False)
        self.convp4b = nn.Conv2d(16, 16, 1, bias=False)
        self.convp4c = nn.Conv2d(16, 16, 1, bias=False)
        self.convp4d = nn.Conv2d(16, 16, 1, bias=False)

        self.prelu = nn.PReLU()

        self.ud0 = UpDown_Module0(r)
        self.ud1 = UpDown_Module(r)
        self.ud2 = UpDown_Module(r)
        self.ud3 = UpDown_Module(r)

        self.con00 = nn.Sequential(
            nn.Conv2d(1, r, 3, 1, 1, bias=True),
            nn.PReLU(),
        )

        self.con00_2 = nn.Sequential(
            nn.Conv2d(1, r, 3, 2, 1, bias=True),
            nn.PReLU(),
        )

        self.con11 = nn.Sequential(
            nn.Conv2d(r+1, r, 3, 1, 1, bias=True),
            nn.PReLU(),
        )

        self.ResB1 = ResB(r)
        self.ResB2 = ResB(r)

        self.up1 = nn.ConvTranspose2d(r, r, kernel_size=5, stride=2, padding=2, output_padding=1)

        self.ConcatConv = ConcatConv(r)

        self.ConvLSTM_l1 = ConvLSTM(r, r, 3)
        self.ConvLSTM_h1 = ConvLSTM(r, r, 3)

        self.ConvLSTM_l2 = ConvLSTM(r, r, 3)
        self.ConvLSTM_h2 = ConvLSTM(r, r, 3)

        self.ConvLSTM_l3 = ConvLSTM(r, r, 3)
        self.ConvLSTM_h3 = ConvLSTM(r, r, 3)

        self.ConvLSTM_l4 = ConvLSTM(r, r, 3)
        self.ConvLSTM_h4 = ConvLSTM(r, r, 3)

    def forward(self, img, g, hyper_para):
        ###################################################################
        # HYP Network
        ###################################################################
        p1 = self.convp1(hyper_para)
        p2 = self.convp2(p1)
        p3 = self.convp3(p2)
        p4a = self.convp4a(p3)
        p4b = self.convp4b(p3)
        p4c = self.convp4c(p3)
        p4d = self.convp4d(p3)

        ###################################################################
        # ROA Network
        ###################################################################
        fg = self.con00(g)
        x2 = self.con11(torch.cat([img, fg], 1))

        # Recurrent-Octave Convolution Layer---1
        self.a_1 = self.ud0(x2)
        self.a_l1 = self.a_1[0]*p4a
        self.a_h1 = self.a_1[1]*p4a
        self.a_l1, self.a_h1 = self.ResB1(self.a_l1), self.ResB2(self.a_h1)
        self.a_l1, h_l1, c_l1 = self.ConvLSTM_l1(self.a_l1, h=None, c=None)
        self.a_h1, h_h1, c_h1 = self.ConvLSTM_h1(self.a_h1, h=None, c=None)

        # Recurrent-Octave Convolution Layer---2
        self.a_2 = self.ud1(self.a_l1, self.a_h1)
        self.a_l2 = self.a_2[0]*p4b
        self.a_h2 = self.a_2[1]*p4b
        self.a_l2, self.a_h2 = self.ResB1(self.a_l2), self.ResB2(self.a_h2)
        self.a_l2, h_l2, c_l2 = self.ConvLSTM_l2(self.a_l2, h_l1, c_l1)
        self.a_h2, h_h2, c_h2 = self.ConvLSTM_h2(self.a_h2, h_h1, c_h1)

        # Recurrent-Octave Convolution Layer---3
        self.a_3 = self.ud2(self.a_l2, self.a_h2)
        self.a_l3 = self.a_3[0]*p4c
        self.a_h3 = self.a_3[1]*p4c
        self.a_l3, self.a_h3 = self.ResB1(self.a_l3), self.ResB2(self.a_h3)
        self.a_l3, h_l3, c_l3 = self.ConvLSTM_l3(self.a_l3, h_l2, c_l2)
        self.a_h3, h_h3, c_h3 = self.ConvLSTM_h3(self.a_h3, h_h2, c_h2)

        # Recurrent-Octave Convolution Layer---4
        self.a_4 = self.ud3(self.a_l3, self.a_h3)
        self.a_l4 = self.a_4[0]*p4d
        self.a_h4 = self.a_4[1]*p4d
        self.a_l4, self.a_h4 = self.ResB1(self.a_l4), self.ResB2(self.a_h4)
        self.a_l4, _, _ = self.ConvLSTM_l4(self.a_l4, h_l3, c_l3)
        self.a_h4, _, _ = self.ConvLSTM_h4(self.a_h4, h_h3, c_h3)

        # Transposed Conv
        self.a_l4_h = self.up1(self.a_l4)

        # ConcatConv
        output = self.ConcatConv(self.a_l4_h, self.a_h4) + self.a_h1

        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 16, 3, padding=0, bias=False), # in_channels, out_channels, kernel_size
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 1, 3, padding=0, bias=False),  # in_channels, out_channels, kernel_size
            nn.Sigmoid()
            )
    def forward(self, im):
        return self.decoder(im)
