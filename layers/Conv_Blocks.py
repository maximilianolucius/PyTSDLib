# Reference: https://openreview.net/pdf?id=ju_Uqw384Oq

import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    """
    Inception_Block_V1 implements an inception-style convolutional block designed to capture multi-scale features.
    It applies multiple convolutional filters with varying kernel sizes in parallel and aggregates their outputs.

    Args:
        in_channels (int): Number of input channels/features.
        out_channels (int): Number of output channels/features after convolution.
        num_kernels (int, optional): Number of different kernel sizes to apply. Default is 6.
        init_weight (bool, optional): Whether to initialize the convolutional weights. Default is True.
    """

    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        # Create a list to hold convolutional layers with varying kernel sizes
        kernels = []
        for i in range(self.num_kernels):
            # Kernel sizes: 1, 3, 5, ..., (2*num_kernels - 1)
            kernel_size = 2 * i + 1
            padding = i  # To maintain the input dimension (same padding)
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            kernels.append(conv)

        # Register the convolutional layers as a ModuleList
        self.kernels = nn.ModuleList(kernels)

        # Initialize weights if specified
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights of convolutional layers using Kaiming Normal initialization.
        Biases are initialized to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Normal initialization for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the Inception_Block_V1.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Channels, Height, Width].

        Returns:
            torch.Tensor: Output tensor after applying multi-scale convolutions and averaging.
                          Shape remains [Batch, out_channels, Height, Width].
        """
        res_list = []
        for i in range(self.num_kernels):
            # Apply each convolutional layer to the input
            res = self.kernels[i](x)
            res_list.append(res)
        # Stack the results along a new dimension and compute the mean across that dimension
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    """
    Inception_Block_V2 extends the inception-style block by incorporating asymmetric convolutions and a 1x1 convolution.
    This design aims to capture more diverse feature representations with fewer parameters.

    Args:
        in_channels (int): Number of input channels/features.
        out_channels (int): Number of output channels/features after convolution.
        num_kernels (int, optional): Number of different kernel sizes to apply. Default is 6.
        init_weight (bool, optional): Whether to initialize the convolutional weights. Default is True.
    """

    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        # Create a list to hold asymmetric convolutional layers
        kernels = []
        for i in range(self.num_kernels // 2):
            # Asymmetric kernel sizes: [1, 3], [1, 5], ..., [1, (2*(num_kernels//2)+1)]
            kernel_size_h = 1
            kernel_size_w = 2 * i + 3
            padding_h = 0
            padding_w = i + 1
            conv_h = nn.Conv2d(in_channels, out_channels, kernel_size=[kernel_size_h, kernel_size_w],
                               padding=[padding_h, padding_w])
            conv_w = nn.Conv2d(in_channels, out_channels, kernel_size=[kernel_size_w, kernel_size_h],
                               padding=[padding_w, padding_h])
            kernels.extend([conv_h, conv_w])

        # Add a 1x1 convolution for dimensionality reduction and feature aggregation
        conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        kernels.append(conv_1x1)

        # Register the convolutional layers as a ModuleList
        self.kernels = nn.ModuleList(kernels)

        # Initialize weights if specified
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights of convolutional layers using Kaiming Normal initialization.
        Biases are initialized to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Normal initialization for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the Inception_Block_V2.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Channels, Height, Width].

        Returns:
            torch.Tensor: Output tensor after applying asymmetric convolutions and averaging.
                          Shape remains [Batch, out_channels, Height, Width].
        """
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            # Apply each convolutional layer to the input
            res = self.kernels[i](x)
            res_list.append(res)
        # Stack the results along a new dimension and compute the mean across that dimension
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
