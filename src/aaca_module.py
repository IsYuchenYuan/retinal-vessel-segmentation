import torch
from torch import nn
import math
from torch.nn.parameter import Parameter

class aaca_layer(nn.Module):
    """Constructs a AACA module.

    Args:
        channel: Number of channels of the input feature map
        gamma, q: two constants, which indicate the slope and the y-intercept of the linear function
                  in a mapping between the channel dimension and kernel size
        lamda, m: two constants, which indicate the slope and the y-intercept of the linear function
                  in a mapping between the dilated rate and kernel size
    """

    def __init__(self, channel,gamma=2, q=1, lamda=2, m=1):
        super(aaca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + q) / gamma))
        ## k: kernel size  ##
        k = t if t % 2 else t - 1
        # r: dilated rate ##
        r = int((k + m) / lamda)

        self.conv = nn.Conv1d(1, 1, kernel_size=k, dilation=r, padding=int((k - 1) * r / 2), bias=False)
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(1, 1, kernel_size=k, dilation=r, padding=int((k - 1) * r / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        ## feature descriptor on the global spatial information
        y_avg = self.avg_pool(x)      
        y = y_avg

        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.conv_1(self.relu(y))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


