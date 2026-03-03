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
"""Multi-Layer Perceptron classifier for tabular malware detection."""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Fully-connected MLP for binary malware classification.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : tuple of int
        Size of each hidden layer.
    output_dim : int
        Number of output classes (2 for binary).
    dropout : float
        Dropout probability applied after each hidden activation.
    """

    def __init__(self, input_dim: int, hidden_dims=(256, 128), output_dim: int = 2, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
