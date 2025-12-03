import torch
import torch.nn as nn
from .layers.fcn_module import FCNModule
from .layers.inception import InceptionModule


class LITE(nn.Module):
    def __init__(
        self,
        n_classes,
        n_filters,
        kernel_size=41,
        use_custom_filters=True,
        use_dilation=True,
        output_directory="",
        dilatation_rate=1,
        device="cpu",
    ):
        super(LITE, self).__init__()

        self.n_classes = n_classes
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.use_custom_filters = use_custom_filters
        self.use_dilation = use_dilation
        self.output_directory = output_directory
        self.dilatation_rate = dilatation_rate

        self.inception = InceptionModule(
            input_channels=1,
            n_filters=self.n_filters,
            dilation_rate=1,
            use_hybrid_layer=self.use_custom_filters,
            device=device,
        )
        self.inception.to(device)

        self.fcn_modules = nn.ModuleList()
        if self.use_custom_filters:
            input_channels = n_filters * 3 + 17
        else:
            input_channels = n_filters * 3
        i = 0
        dilation_rate = 2 ** (i + 1) if self.use_dilation else 1
        fcn_module1 = FCNModule(
            input_channels=input_channels,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size // (2**i),
            dilation_rate=dilation_rate,
            stride=1,
            activation="relu",
            device=device,
        )
        fcn_module1.to(device)
        self.fcn_modules.append(fcn_module1)
        i = 1
        dilation_rate = 2 ** (i + 1) if self.use_dilation else 1
        fcn_module2 = FCNModule(
            input_channels=n_filters,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size // (2**i),
            dilation_rate=dilation_rate,
            stride=1,
            activation="relu",
            device=device,
        )
        fcn_module2.to(device)

        self.fcn_modules.append(fcn_module2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1).to(device)

        self.output_layer = nn.Linear(self.n_filters, self.n_classes).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, x):
        x = x.view(-1, 1, x.shape[-1])
        x = self.inception(x)
        for fcn_module in self.fcn_modules:
            x = fcn_module(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x
