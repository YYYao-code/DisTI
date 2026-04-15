
import os,pickle
import numpy as np
import torch
import torch.nn as nn

import torch
import torch.nn.functional as F


class Chomp1d(torch.nn.Module):
    """Removes leading or trailing elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    Args:
        chomp_size : Number of elements to remove.
    """

    def __init__(self, chomp_size: int, last: bool = True):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size]


class TCNBlock(torch.nn.Module):
    """Temporal Convolutional Network block.

    Composed sequentially of two causal convolutions (with leaky ReLU activation functions),
    and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    Args:
        in_channels : Number of input channels.
        out_channels : Number of output channels.
        kernel_size : Kernel size of the applied non-residual convolutions.
        dilation : Dilation parameter of non-residual convolutions.
        final : If True, the last activation function is disabled.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
            groups:int,
            final: bool = False,
    ):

        super(TCNBlock, self).__init__()

        in_channels = int(in_channels)
        kernel_size = int(kernel_size)
        out_channels = int(out_channels)
        dilation = int(dilation)
        groups = int(groups)

        # Computes left padding so that the applied convolutions are causal
        padding = int((kernel_size - 1) * dilation)

        # First causal convolution
        conv1_pre = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        conv1 = torch.nn.utils.weight_norm(conv1_pre)

        # The truncation makes the convolution causal
        chomp1 = Chomp1d(chomp_size=padding)

        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2_pre = torch.nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        conv2 = torch.nn.utils.weight_norm(conv2_pre)
        chomp2 = Chomp1d(chomp_size=padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(conv1, chomp1, relu1, conv2, chomp2, relu2)

        # Residual connection
        self.upordownsample = (
            torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        # Final activation function
        self.activation = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.activation is None:
            return out_causal + res
        else:
            return self.activation(out_causal + res)


class TCN(torch.nn.Module):
    """Temporal Convolutional Network.

    Composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    Args:
        in_channels : Number of input channels.
        out_channels : Number of output channels.
        kernel_size : Kernel size of the applied non-residual convolutions.
        channels : Number of channels processed in the network and of output
            channels.
        layers : Depth of the network.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            channels: int,
            layers: int,
            groups:int
    ):
        super(TCN, self).__init__()

        layers = int(layers)

        net_layers = []  # List of sequential TCN blocks
        dilation_size = 1  # Initial dilation size

        for i in range(layers):
            in_channels_block = in_channels if i == 0 else channels
            net_layers.append(
                TCNBlock(
                    in_channels=in_channels_block,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    groups=groups,
                    final=False,
                )
            )
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        net_layers.append(
            TCNBlock(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation_size,
                groups=groups,
                final=True,
            )
        )

        self.network = torch.nn.Sequential(*net_layers)

    def forward(self, x):
        return self.network(x)





#参考BeatGAN, 参数设置中z的channel有压缩; encoder, decoder中间层channel所倍乘的基准ngf, ndf相等
#vconv先经过channel变换后和t长度一直，故可取相同的参数，避免不同数据集v不同需要对conv设置不同的参数。 （是channel变换，不是embedding）

# ConvBNRelu :in_channels, out_channels, kernel_size=3, stride=1, groups=1
class Encoder(nn.Module):
    def __init__(self,d_model,c_in,window):
        super(Encoder, self).__init__()
        # self.ngpu = ngpu
        self.c_in = c_in
        self.window=window
        self.d_model = d_model
        self.tconv = TCN(d_model,d_model,7,d_model,5,groups=d_model)
        self.vconv = TCN(window,window,7,window,5,groups=window)
        #self.merge=nn.Conv1d(nz, nz, 1, 1, bias=False) #考虑前后是否需要加activation
        # self.t_rec=TCN(d_model,d_model,7,d_model,3,groups=d_model)
        self.t_rec=nn.Sequential(
            TCN(d_model,d_model,7,d_model,2,groups=d_model),
            nn.Conv1d(d_model,c_in,kernel_size=1,bias=False),
        )
        # self.v_rec=TCN(window,window,7,window,3,groups=window)
        self.v_rec1=TCN(window, window, 7, window, 2, groups=window)
        self.v_rec2=nn.Conv1d(d_model, c_in, 1,1,0, bias=False)

        self.merge=nn.Conv1d(2*d_model,d_model,1,1,0,bias=False)
        self.reconstruct=nn.Linear(d_model*window,c_in*window)
    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:

        t_feature=self.tconv(input)
        v_feature=self.vconv(input.transpose(1,2)).transpose(1,2) #和t_feature保持相同的shape
        combined1=torch.multiply(t_feature,nn.Softmax(dim=1)(v_feature))
        combined2=torch.multiply(nn.Softmax(dim=-1)(t_feature),v_feature)
        fusion=self.merge(torch.concat([combined1,combined2],dim=1))
        result=self.reconstruct(fusion.flatten(start_dim=1)).reshape(-1, self.c_in, self.window)
        r_tt = self.t_rec(t_feature)
        r_tv = self.t_rec(v_feature)
        r_vv = self.v_rec2(self.v_rec1(v_feature.transpose(1,2)).transpose(1,2))  #由于采用mlp,此处v_feature,t_feature不用transpose,直接flatten即可 #.transpose(1, 2)
        r_vt = self.v_rec2(self.v_rec1(t_feature.transpose(1,2)).transpose(1,2)) #.transpose(1, 2)

        return result,fusion,r_tt,r_tv,r_vv,r_vt,t_feature,v_feature



##

#注意decoder在时间维度复原结果要和encoder一致。为层数统一，设为100-50-25-5
class Decoder(nn.Module):
    def __init__(self, c_out,nc,nz):
        super(Decoder, self).__init__()
        # self.ngpu = ngpu
        self.main=nn.Sequential(

            nn.ConvTranspose1d(nz, nc*2, 7, 5, 1, bias=False),
            nn.BatchNorm1d(nc*2),
            nn.ReLU(True),
            # state size. (ngf) x 80
            nn.ConvTranspose1d(nc * 2, nc, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nc),
            nn.ReLU(True),
            # state size. (ngf) x 160
            nn.ConvTranspose1d(nc, c_out, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 320

        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)
        return output

#此处不仅是channel 数目变换，亦同时进行特征提取 参考mobilev2，其在堆叠bottleneck前首层conv取k=3
#主要还是channel等形式统一，特征提取次要，故k=1，且只用conv1d
#v维度：长度和window统一，channel变为指定维度。t维度：channel变为指定维度。
class Transform(nn.Module):
    def __init__(self, c_in,window,d_model):
        super(Transform, self).__init__()
        # self.ngpu = ngpu
        self.c_in = c_in
        # self.window=window
        # self.ttrans = ConvBNReLU(c_in, d_model,3)
        # self.vtrans = ConvBNReLU(window, d_model,3)
        # self.ttrans=nn.Sequential(
        #     ConvBNReLU(c_in,d_model,1),
        #     ConvBNReLU(d_model,d_model,1),
        # )
        # self.vlayer1=nn.Conv1d(c_in,window,1)#在欲处理的长度上和t维度的window保持一致,后方conv1d在参数上不必根据dataset变化
        #                                                     # #ConvBNReLU(c_in,d_model,1)
        # self.vlayer2=nn.Conv1d(window,d_model,1)#channel变成指定数目#ConvBNReLU(window,c_in,1)
        self.tlayer1=nn.Conv1d(c_in,d_model,1)  #channel变成指定数目    #ConvBNReLU(window,d_model,1)

    def forward(self, input):
        # t_out=self.ttrans(input)
        # v_out=self.vtrans(input.transpose(1,2))
        # v_out=self.vlayer2(self.vlayer1(input).transpose(1,2))
        # t_out=self.tlayer1(input)
        output=self.tlayer1(input)
        return output



#mobilenetV2中除了堆叠的bottleneck以外，首末conv均采用ConvBNRelu,故channel transform也需要激活函数
#对t,v均采用channel transform
class Reconstructor(nn.Module):
    def __init__(self,c_in,c_out,nc,nz,d_model,window,num_kernels):
        super(Reconstructor, self).__init__()

        # Encoder
        self.encoder = Encoder(d_model,nc,nz,c_in,window,num_kernels)
        # self.decoder = Decoder(c_out,nc,nz)
        self.transform=Transform(c_in,window,d_model)



    def forward(self, x):
        input_emb=self.transform(x)
        result,combined,r_tt,r_tv,r_vv,r_vt,t_feature,v_feature=self.encoder(input_emb)
        # output=self.decoder(z)

        return result,combined,r_tt,r_tv,r_vv,r_vt,t_feature,v_feature

