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
"""Graph Convolutional Network classifier for tabular malware detection.

Pure-PyTorch implementation — no PyG or DGL dependency required.
The caller is responsible for passing a pre-computed, symmetrically
normalised adjacency matrix (D^{-1/2} A_hat D^{-1/2} where A_hat = A + I).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Single graph-convolutional layer: H' = σ(A_norm · H · W).

    Parameters
    ----------
    in_features : int
        Number of input node features.
    out_features : int
        Number of output node features.
    bias : bool
        Whether to add a learnable bias term.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (N, in_features)
            Node feature matrix.
        adj_norm : Tensor of shape (N, N)
            Symmetrically normalised adjacency matrix (dense).

        Returns
        -------
        Tensor of shape (N, out_features)
        """
        support = x @ self.weight          # (N, out_features)
        out = adj_norm @ support            # (N, out_features)
        if self.bias is not None:
            out = out + self.bias
        return out


class GCNClassifier(nn.Module):
    """Two-layer GCN for node-level binary malware classification.

    Parameters
    ----------
    input_dim : int
        Number of input node features.
    hidden_dim : int
        Number of hidden units in the first GCN layer.
    output_dim : int
        Number of output classes (2 for binary).
    dropout : float
        Dropout probability applied after the first hidden activation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 2, dropout: float = 0.5):
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (N, input_dim)
        adj_norm : Tensor of shape (N, N)

        Returns
        -------
        Tensor of shape (N, output_dim) — raw logits.
        """
        h = F.relu(self.gcn1(x, adj_norm))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.gcn2(h, adj_norm)
        return out
