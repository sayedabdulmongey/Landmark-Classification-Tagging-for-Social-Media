import torch
import torch.nn as nn
from collections import OrderedDict

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("conv_0", get_conv_layer(in_channels=3,  out_channels=64)),
                    ("conv_1", get_conv_layer(in_channels=64, out_channels=64)),
                    ("pool_0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)), 

                    ("conv_2", get_conv_layer(in_channels=64, out_channels=128)),
                    ("conv_3", get_conv_layer(in_channels=128, out_channels=128)),
                    ("pool_1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)), 

                    ("conv_4", get_conv_layer(in_channels=128, out_channels=256)),
                    ("conv_5", get_conv_layer(in_channels=256, out_channels=256)),
                    ("conv_6", get_conv_layer(in_channels=256, out_channels=256)),
                    ("pool_2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

                    ("conv_7", get_conv_layer(in_channels=256, out_channels=512)),
                    ("conv_8", get_conv_layer(in_channels=512, out_channels=512)),
                    ("conv_9", get_conv_layer(in_channels=512, out_channels=512)),
                    ("pool_3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

                    ("res_block_0", ResidualBlock(in_channels=512, out_channels=512)),
                    ("res_block_1", ResidualBlock(in_channels=512, out_channels=512)),
                    
                    ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
                    ("flatten", nn.Flatten()),
                    ("fc", get_fc_layer(512, 1024, dropout=dropout)),
                    ("fc_2", nn.Linear(1024, num_classes))
                ]
            )
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)
    
def get_fc_layer(in_features, out_features,dropout=0.5):
    fc_layer = nn.Sequential(
        OrderedDict(
            [
                ("fc", nn.Linear(in_features, out_features)),
                ("dropout", nn.Dropout(dropout)),
                ("relu", nn.ReLU()),
            ]
        )
    )
    return fc_layer
    
def get_conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    conv_layer = nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
                ("bn", nn.BatchNorm2d(out_channels)),
                ("relu", nn.ReLU()),
            ]
        )
    )
    return conv_layer



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):

            super().__init__()

            self.conf_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1", nn.Sequential(
                            OrderedDict(
                                [
                                    ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
                                    ("bn", nn.BatchNorm2d(out_channels)),
                                    ("relu", nn.ReLU()),
                                ]
                            )
                        )
                    ),
                    (
                        "conv2", nn.Sequential(
                            OrderedDict(
                                [
                                    ("conv", nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                                    ("bn", nn.BatchNorm2d(in_channels)),
                                ]
                        )
                    )
                    )
                ]
            )
            )
            if stride != 1 or in_channels != out_channels:
                  self.downsample = nn.Sequential(
                      OrderedDict(
                          [
                              ('downsample_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                              ('downsample_bn', nn.BatchNorm2d(out_channels)),
                          ]
                      )
                  )
            else :
                  self.downsample = None
            self.relu = nn.ReLU()
    def forward(self, x):
            # F
            F = self.conf_block(x)
            if self.downsample is not None:
                   residual = self.downsample(x)
            else :
                  residual = x
            H = F + residual
            return self.relu(H)
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
