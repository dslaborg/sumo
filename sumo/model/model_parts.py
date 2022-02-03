from typing import Union

import torch
import torch.nn as nn


class SingleConv(nn.Module):
    """
    Module with a 1D convolution with the given parameters, followed by an activation function and batch normalization.

    Parameters
    ----------
    in_channels : int
        The number of input channels of the convolution.
    out_channels : int
        The number of channels produced by the convolution.
    activation : nn.Module
        One of the activation functions as defined in pytorch.
    kernel_size : int
        Width of the kernel of the convolution.
    padding : Union[str, int]
        Padding added to both sided of the convolution input; can be `'valid'` or `'same'` (see pytorch documentation)
        or an integer.
    dilation : int
        Spacing between kernel elements of the convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, activation: nn.Module, kernel_size: int,
                 padding: Union[str, int], dilation: int):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            activation,
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.single_conv(x)


class DoubleConv(nn.Module):
    """
    Module with two consecutive `SingleConv` modules with the same amount of output channels for both convolutions.

    Parameters
    ----------
    in_channels : int
        The number of input channels of the first convolution.
    out_channels : int
        The number of channels produced by the second convolution (and implicitly also by the first convolution).
    **kwargs : dict
        The remaining arguments passed to the `SingleConv` module (activation function and convolution parameters).
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs: dict):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            SingleConv(in_channels, out_channels, **kwargs),
            SingleConv(out_channels, out_channels, **kwargs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Encoder(nn.Module):
    """
    Module representing a branching encoder used "on the left side" of the U.

    It contains a max pooling operation, followed by a `DoubleConv` module.

    Parameters
    ----------
    in_channels : int
        The number of input channels of the first convolution.
    out_channels : int
        The number of channels produced by the second convolution (and implicitly also by the first convolution).
    pool : int
        Width of the sliding windows used by the max pooling module.
    **kwargs : dict
        The remaining arguments passed to the `kwargs` parameter of the `DoubleConv` module.
    """

    def __init__(self, in_channels: int, out_channels: int, pool: int, **kwargs: dict):
        super(Encoder, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(pool),
            DoubleConv(in_channels, out_channels, **kwargs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Decoder(nn.Module):
    """
    Module representing a converging decoder used "on the right side" of the U.

    It contains an upsampling operation, followed by a `SingleConv` module. Afterwards the output is appended to the
    output of the corresponding encoder (in the channel dimension, skip connection) and the result fed into a
    `DoubleConv` module.

    Parameters
    ----------
    in_channels : int
        The number of input channels of the first convolution.
    out_channels : int
        The number of channels produced by the second convolution (and implicitly also by the first convolution).
    scale_factor : int
        Multiplier for temporal size used by the upsampling module.
    **kwargs : dict
        The remaining arguments passed to the `kwargs` parameter of the `DoubleConv` module.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int, **kwargs: dict):
        super(Decoder, self).__init__()

        kwargs_single_conv = kwargs.copy()
        # use scale_factor also as kernel size in single convolution
        kwargs_single_conv['kernel_size'] = scale_factor

        self.up = nn.Upsample(scale_factor=scale_factor)
        self.single_conv = SingleConv(in_channels, out_channels, **kwargs_single_conv)
        self.double_conv = DoubleConv(in_channels, out_channels, **kwargs)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # upsample the output data of the previous decoder
        x1 = self.up(x1)
        # feed the upsampled tensor into a single convolution, which halves the number of channels
        x1 = self.single_conv(x1)

        assert x1.shape[-1] == x2.shape[-1]
        # concatenate the output of the corresponding encoder with the result of the `SingleConv` module, which doubles
        # the number of channels again
        x = torch.cat([x2, x1], dim=1)

        # feed the concatenated tensor into a double convolution, which halves the number of channels once more
        return self.double_conv(x)
