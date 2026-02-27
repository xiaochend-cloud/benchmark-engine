# Copyright (c) 2022-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ResidualAEModule — AEModule with skip connections in encoder and decoder."""

import torch

from morpheus.models.dfencoder.ae_module import AEModule
from morpheus.models.dfencoder.ae_module import CompleteLayer


class ResidualBlock(torch.nn.Module):
    """Wraps a CompleteLayer with an optional residual (skip) connection.

    If the input and output dimensions match the connection is additive.
    Otherwise a learned linear projection is used to match dimensions.
    """

    def __init__(self, in_dim: int, out_dim: int, activation=None, dropout=None):
        super().__init__()
        self.layer = CompleteLayer(in_dim, out_dim, activation=activation, dropout=dropout)
        if in_dim != out_dim:
            self.projection = torch.nn.Linear(in_dim, out_dim)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.projection is None else self.projection(x)
        return self.layer(x) + residual


class ResidualAEModule(AEModule):
    """Auto Encoder module with residual (skip) connections in each encoder/decoder layer."""

    def _build_layers(self, input_dim: int) -> int:
        """Build encoder and decoder layers, each wrapped in a ResidualBlock.

        Parameters
        ----------
        input_dim : int
            The input dimension to the first encoder layer.

        Returns
        -------
        int
            The output dimension of the final decoder layer (or encoder if no decoder).
        """
        # Fill in defaults the same way the parent does
        if self.encoder_layers is None:
            self.encoder_layers = [int(4 * input_dim) for _ in range(3)]

        if self.decoder_layers is None:
            self.decoder_layers = []

        if self.encoder_activations is None:
            self.encoder_activations = [self.activation for _ in self.encoder_layers]

        if self.decoder_activations is None:
            self.decoder_activations = [self.activation for _ in self.decoder_layers]

        if self.encoder_dropout is None or type(self.encoder_dropout) == float:
            drp = self.encoder_dropout
            self.encoder_dropout = [drp for _ in self.encoder_layers]

        if self.decoder_dropout is None or type(self.decoder_dropout) == float:
            drp = self.decoder_dropout
            self.decoder_dropout = [drp for _ in self.decoder_layers]

        for i, dim in enumerate(self.encoder_layers):
            block = ResidualBlock(input_dim, dim, activation=self.encoder_activations[i],
                                  dropout=self.encoder_dropout[i])
            input_dim = dim
            self.encoder.append(block)
            self.add_module(f"encoder_{i}", block)

        for i, dim in enumerate(self.decoder_layers):
            block = ResidualBlock(input_dim, dim, activation=self.decoder_activations[i],
                                  dropout=self.decoder_dropout[i])
            input_dim = dim
            self.decoder.append(block)
            self.add_module(f"decoder_{i}", block)

        return input_dim

    def forward(self, input: torch.Tensor):
        """Forward pass: encode then decode."""
        encoding = self.encode(input)
        num, bin, cat = self.decode(encoding)
        return num, bin, cat
