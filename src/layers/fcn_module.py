import torch
import torch.nn as nn


class SeparableConv1D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, device="cpu"
    ):
        super(SeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding="same",
            dilation=dilation,
            groups=in_channels,
            bias=False,
        ).to(device)
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            1,
            stride=1,
            padding="same",
            dilation=1,
            groups=1,
            bias=False,
        ).to(device)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class FCNModule(nn.Module):
    def __init__(
        self,
        input_channels,
        n_filters,
        kernel_size,
        dilation_rate=1,
        stride=1,
        activation="relu",
        device="cpu",
    ):
        super(FCNModule, self).__init__()
        self.separable_conv = SeparableConv1D(
            input_channels,
            n_filters,
            kernel_size,
            stride=stride,
            dilation=dilation_rate,
            device=device,
        ).to(device)
        self.batch_norm = nn.BatchNorm1d(n_filters).to(device)
        self.activation = (
            nn.ReLU().to(device) if activation == "relu" else nn.Identity()
        )

    def forward(self, input_tensor):
        x = self.separable_conv(input_tensor)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x
