import torch
import torch.nn as nn
import numpy as np


class HybridLayer(nn.Module):
    """Layer création de layer "imposé"

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, input_channels, kernel_sizes=[2, 4, 8, 16, 32, 64], device="cpu"
    ):
        super(HybridLayer, self).__init__()
        self.input_channels = input_channels
        self.kernel_sizes = kernel_sizes
        self.conv_layers = nn.ModuleList()

        for kernel_size in kernel_sizes:
            filter_ = np.ones((kernel_size, input_channels, 1))
            indices_ = np.arange(kernel_size)
            filter_[indices_ % 2 == 0] *= -1
            filter_ = torch.tensor(filter_, dtype=torch.float32).permute(2, 1, 0)
            conv_layer = nn.Conv1d(
                input_channels, 1, kernel_size, padding="same", bias=False
            ).to(device)
            with torch.no_grad():
                conv_layer.weight.copy_(filter_)
            conv_layer.weight.requires_grad = False
            self.conv_layers.append(conv_layer)

        for kernel_size in kernel_sizes:
            filter_ = np.ones((kernel_size, input_channels, 1))
            indices_ = np.arange(kernel_size)
            filter_[indices_ % 2 > 0] *= -1
            filter_ = torch.tensor(filter_, dtype=torch.float32).permute(2, 1, 0)
            conv_layer = nn.Conv1d(
                input_channels, 1, kernel_size, padding="same", bias=False
            ).to(device)
            with torch.no_grad():
                conv_layer.weight.copy_(filter_)
            conv_layer.weight.requires_grad = False
            self.conv_layers.append(conv_layer)

        for kernel_size in kernel_sizes[1:]:
            filter_ = np.zeros((kernel_size + kernel_size // 2, input_channels, 1))
            xmash = np.linspace(0, 1, kernel_size // 4 + 1)[1:].reshape((-1, 1, 1))
            filter_left = xmash**2
            filter_right = filter_left[::-1]
            filter_[0 : kernel_size // 4] = -filter_left
            filter_[kernel_size // 4 : kernel_size // 2] = -filter_right
            filter_[kernel_size // 2 : 3 * kernel_size // 4] = 2 * filter_left
            filter_[3 * kernel_size // 4 : kernel_size] = 2 * filter_right
            filter_[kernel_size : 5 * kernel_size // 4] = -filter_left
            filter_[5 * kernel_size // 4 :] = -filter_right
            filter_ = torch.tensor(filter_, dtype=torch.float32).permute(2, 1, 0)
            conv_layer = nn.Conv1d(
                input_channels,
                1,
                kernel_size + kernel_size // 2,
                padding="same",
                bias=False,
            ).to(device)
            with torch.no_grad():
                conv_layer.weight.copy_(filter_)
            conv_layer.weight.requires_grad = False
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU().to(device)

    def forward(self, input_tensor):
        conv_list = [conv(input_tensor) for conv in self.conv_layers]
        hybrid_layer = torch.cat(conv_list, dim=1)
        hybrid_layer = self.relu(hybrid_layer)
        return hybrid_layer  # .permute(0,2,1)
