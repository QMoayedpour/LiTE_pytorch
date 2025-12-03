import torch
import torch.nn as nn
from .hybrid_layer import HybridLayer


class InceptionModule(nn.Module):
    def __init__(
        self,
        input_channels,
        n_filters,
        dilation_rate,
        stride=1,
        kernel_size=41,
        activation="linear",
        use_hybrid_layer=False,
        use_multiplexing=True,
        device="cpu",
    ):
        super(InceptionModule, self).__init__()
        self.n_filters = n_filters
        self.dilation_rate = dilation_rate
        self.stride = stride
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_hybrid_layer = use_hybrid_layer
        self.use_multiplexing = use_multiplexing
        self.device = device

        if not self.use_multiplexing:
            self.n_convs = 1
            self.n_filters *= 3
        else:
            self.n_convs = 3

        self.kernel_size_s = [self.kernel_size // (2**i) for i in range(self.n_convs)]
        self.conv_list = nn.ModuleList()

        for i in range(len(self.kernel_size_s)):
            conv = nn.Conv1d(
                input_channels,
                self.n_filters,
                self.kernel_size_s[i],
                stride=self.stride,
                padding="same",
                dilation=self.dilation_rate,
                bias=False,
            ).to(device)
            self.conv_list.append(conv)

        if self.use_hybrid_layer:
            n = self.n_filters * self.n_convs + 17
        else:
            n = self.n_filters * self.n_convs
        self.batch_norm = nn.BatchNorm1d(n).to(device)
        self.relu = nn.ReLU().to(device)

    def forward(self, input_tensor):
        conv_outputs = [conv(input_tensor) for conv in self.conv_list]
        if self.use_hybrid_layer:
            self.hybrid = HybridLayer(
                input_channels=input_tensor.shape[1], device=self.device
            ).to(self.device)
            hybrid_output = self.hybrid(input_tensor)
            conv_outputs.append(hybrid_output)
        if len(conv_outputs) > 1:
            concatenated = torch.cat(conv_outputs, dim=1)
        else:
            concatenated = conv_outputs[0]

        x = self.batch_norm(concatenated)
        x = self.batch_norm(concatenated)
        x = self.relu(x)

        return x
