# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)

# ---------------------------------------------------------------------------
# Custom Models
# ---------------------------------------------------------------------------

class CNNLSTMEncoder(nn.Module):
    """A convolutional LSTM based sequence encoder.
    
    Applies dilated 1D convolutions for expansive local feature extraction 
    over time, followed by a Bidirectional LSTM. Uses 'same' padding to 
    ensure the temporal dimension (T) remains unchanged for CTC loss.

    Args:
        num_features (int): `num_features` for an input of shape (T, N, num_features).
        cnn_channels (list): A list of output channels per 1D convolutional layer.
        cnn_kernel_sizes (list): A list of kernel sizes for the convolutions.
        cnn_dilations (list): A list of dilation rates for the convolutions.
        rnn_hidden_size (int): The hidden size for the BiLSTM.
        rnn_num_layers (int): The number of stacked BiLSTM layers.
        rnn_dropout (float): Dropout probability for the BiLSTM.
    """

    def __init__(
        self,
        num_features: int,
        cnn_channels: Sequence[int] = (64, 128, 256),
        cnn_kernel_sizes: Sequence[int] = (15, 15, 15),
        cnn_dilations: Sequence[int] = (1, 2, 4),
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        assert len(cnn_channels) == len(cnn_kernel_sizes) == len(cnn_dilations), "Mismatched CNN config"

        cnn_blocks: list[nn.Module] = []
        in_channels = num_features
        
        for out_channels, k_size, dilation in zip(cnn_channels, cnn_kernel_sizes, cnn_dilations):
            cnn_blocks.extend(
                [
                    nn.Conv1d(
                        in_channels, 
                        out_channels, 
                        kernel_size=k_size, 
                        dilation=dilation,
                        padding="same"
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_blocks)
        
        # LayerNorm helps stabilize the inputs before feeding them into the unrolled RNN
        self.pre_rnn_norm = nn.LayerNorm(in_channels)

        self.rnn = nn.LSTM(
            input_size=in_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=True,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
        )
        
        self.out_features = rnn_hidden_size * 2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        
        # Conv1d expects (N, C, T)
        x = inputs.movedim(0, -1)
        x = self.cnn(x)
        
        # LSTM expects (T, N, C_out)
        x = x.movedim(-1, 0)
        x = self.pre_rnn_norm(x)
        outputs, _ = self.rnn(x)
        
        return outputs  # (T, N, rnn_hidden_size * 2)


class RNNBiLSTMEncoder(nn.Module):
    """An RNN encoder that uses Bi-LSTM.
    
    Args:
        num_features (int): `num_features` for an input of shape (T, N, num_features).
        rnn_hidden_size (int): The hidden size for the BiLSTM.
        rnn_num_layers (int): The number of stacked BiLSTM layers.
        rnn_dropout (float): Dropout probability for the BiLSTM.
    """

    def __init__(
        self,
        num_features: int,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 5,
        rnn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        in_channels = num_features
        self.rnn = nn.LSTM(
            input_size=in_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=True,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
        )
        # self.norm = nn.LayerNorm(rnn_hidden_size * 2)
        
        self.out_features = rnn_hidden_size * 2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        outputs, _ = self.rnn(inputs)
        # outputs = self.norm(outputs)
        
        return outputs  # (T, N, rnn_hidden_size * 2)


class ChannelSubset(nn.Module):
    """Selects the first ``num_channels`` electrode channels from the input.

    Input shape:  (T, N, num_bands, electrode_channels, freq)
    Output shape: (T, N, num_bands, num_channels, freq)

    Args:
        num_channels (int): Number of electrode channels to keep (taken from
            the front of the channel dimension).
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :, :, : self.num_channels, :]


class Conv1DBlock(nn.Module):
    """A 1D convolutional block with BatchNorm, ReLU, and dropout.

    Uses same-length ("same") convolution so the temporal dimension T is
    preserved. Expects inputs of shape (T, N, in_channels) and returns
    (T, N, out_channels).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Temporal kernel size (should be odd for exact
            same-length output).
        dropout (float): Dropout probability applied after activation.
            (default: 0.1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (T, N, in_channels) -> (N, in_channels, T)
        x = inputs.permute(1, 2, 0)
        x = self.dropout(self.relu(self.bn(self.conv(x))))
        # (N, out_channels, T) -> (T, N, out_channels)
        return x.permute(2, 0, 1)


class CNNGRUEncoder(nn.Module):
    """A 1D CNN encoder followed by a bidirectional GRU for sequence modeling.

    Takes an input of shape (T, N, num_features), applies a stack of
    ``Conv1DBlock`` layers along the time axis (preserving T via same-padding),
    then feeds the output through a bidirectional multi-layer GRU.

    Returns a tensor of shape (T, N, gru_hidden_size * 2).

    Args:
        num_features (int): Input feature dimension (i.e. C in (T, N, C)).
        cnn_channels (list): Output channel counts for each Conv1DBlock.
        cnn_kernel_size (int): Temporal kernel size shared by all Conv1DBlocks.
        gru_hidden_size (int): GRU hidden size per direction; output features
            are ``gru_hidden_size * 2`` due to bidirectionality.
        gru_num_layers (int): Number of stacked GRU layers.
        dropout (float): Dropout probability for Conv1DBlocks and between GRU
            layers (only applied when ``gru_num_layers > 1``). (default: 0.1)
    """

    def __init__(
        self,
        num_features: int,
        cnn_channels: Sequence[int],
        cnn_kernel_size: int,
        gru_hidden_size: int,
        gru_num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert len(cnn_channels) > 0
        cnn_blocks: list[nn.Module] = []
        in_c = num_features
        for out_c in cnn_channels:
            cnn_blocks.append(Conv1DBlock(in_c, out_c, cnn_kernel_size, dropout))
            in_c = out_c
        self.cnn = nn.Sequential(*cnn_blocks)

        # Dropout between GRU layers if multiple layers are used
        self.gru = nn.GRU(
            input_size=in_c,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )
        self.out_features = gru_hidden_size * 2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.cnn(inputs)   # (T, N, cnn_channels[-1])
        x, _ = self.gru(x)     # (T, N, gru_hidden_size * 2)
        return x


class CNNEncoder(nn.Module):
    """Pure 1D CNN encoder with dilated convolutions.

    Stacks ``Conv1DBlock`` layers with increasing dilations to achieve a large
    temporal receptive field without a recurrent layer.

    Args:
        num_features (int): Input feature dimension.
        cnn_channels (list): Output channel count for each block.
        cnn_kernel_size (int): Temporal kernel size shared by all blocks.
        cnn_dilations (list): Dilation for each block; same length as
            ``cnn_channels``.
        dropout (float): Dropout probability in each block. (default: 0.1)
    """

    def __init__(
        self,
        num_features: int,
        cnn_channels: Sequence[int],
        cnn_kernel_size: int,
        cnn_dilations: Sequence[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert len(cnn_channels) == len(cnn_dilations), (
            "cnn_channels and cnn_dilations must have the same length"
        )

        blocks: list[nn.Module] = []
        in_c = num_features
        for out_c, dil in zip(cnn_channels, cnn_dilations):
            blocks.append(Conv1DBlock(in_c, out_c, cnn_kernel_size, dropout, dil))
            in_c = out_c
        self.cnn = nn.Sequential(*blocks)
        self.out_features = in_c

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.cnn(inputs)  # (T, N, cnn_channels[-1])

