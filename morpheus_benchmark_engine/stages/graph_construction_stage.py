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
"""Graph construction stage: builds a k-NN adjacency matrix from cosine similarity.

The stage reads ``X_train``, ``X_val``, ``X_test`` from the incoming
ControlMessage metadata, stacks them into a single feature matrix, computes
pairwise cosine similarity, keeps the top-k neighbours per node, and
produces a symmetrically normalised adjacency matrix
``A_norm = D^{-1/2} (A + I) D^{-1/2}``.

Additional metadata added to the outgoing ControlMessage:
  - ``adj_norm``   — float32 numpy array, shape (N_total, N_total)
  - ``n_train``    — int, number of training nodes
  - ``n_val``      — int, number of validation nodes
  - ``n_test``     — int, number of test nodes
"""

import logging
import typing

import mrc
import numpy as np
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus_benchmark_engine.utils import progress_tracker

logger = logging.getLogger(f"morpheus.{__name__}")


class GraphConstructionStage(SinglePortStage):
    """Build a k-NN cosine-similarity graph from the preprocessed feature splits.

    Parameters
    ----------
    c : Config
        Pipeline configuration instance.
    k : int
        Number of nearest neighbours to keep per node (excluding self).
    """

    def __init__(self, c: Config, k: int = 10):
        super().__init__(c)
        self._k = k

    @property
    def name(self) -> str:
        return "graph-construction"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage,)

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(schema.input_type)

    @staticmethod
    def _cosine_similarity(X: np.ndarray) -> np.ndarray:
        """Row-wise cosine similarity matrix for X (N, F) → (N, N)."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
        X_norm = X / norms
        return X_norm @ X_norm.T                    # (N, N)

    @staticmethod
    def _knn_adjacency(sim: np.ndarray, k: int) -> np.ndarray:
        """Retain only the top-k entries per row; set everything else to 0."""
        n = sim.shape[0]
        adj = np.zeros_like(sim)
        # For each node keep the k highest-similarity neighbours (not self)
        for i in range(n):
            row = sim[i].copy()
            row[i] = -np.inf    # exclude self
            top_k_idx = np.argpartition(row, -k)[-k:]
            adj[i, top_k_idx] = sim[i, top_k_idx]
        return adj

    @staticmethod
    def _normalise_adjacency(adj: np.ndarray) -> np.ndarray:
        """Symmetric normalisation: D^{-1/2} (A + I) D^{-1/2}."""
        a_hat = adj + np.eye(adj.shape[0], dtype=adj.dtype)
        d = a_hat.sum(axis=1)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        return (D_inv_sqrt @ a_hat @ D_inv_sqrt).astype(np.float32)

    def on_data(self, message: ControlMessage) -> ControlMessage:
        if message is None:
            return None

        X_train: np.ndarray = message.get_metadata("X_train")
        X_val: np.ndarray = message.get_metadata("X_val")
        X_test: np.ndarray = message.get_metadata("X_test")

        if X_train is None:
            raise RuntimeError("'X_train' not found in message metadata — run MalwarePreprocessingStage first.")

        # Stack all splits into a single matrix for graph construction
        X_all = np.vstack([X_train, X_val, X_test])
        n_total = X_all.shape[0]
        k = min(self._k, n_total - 1)

        logger.info(
            "Building %d-NN cosine-similarity graph for %d nodes (train=%d, val=%d, test=%d)",
            k, n_total, len(X_train), len(X_val), len(X_test)
        )

        progress_tracker.update("graph", status="in_progress", k=k, n_nodes=n_total)
        sim = self._cosine_similarity(X_all)
        adj = self._knn_adjacency(sim, k)
        adj_norm = self._normalise_adjacency(adj)
        progress_tracker.update("graph", status="completed", k=k, n_nodes=n_total)

        out = ControlMessage()
        out.payload(message.payload())
        # Forward all existing split metadata
        for key in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test",
                    "feature_names", "num_features", "split_seed"):
            out.set_metadata(key, message.get_metadata(key))
        # Add graph-specific metadata
        out.set_metadata("adj_norm", adj_norm)
        out.set_metadata("n_train", len(X_train))
        out.set_metadata("n_val", len(X_val))
        out.set_metadata("n_test", len(X_test))

        return out

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data), ops.filter(lambda x: x is not None))
        builder.make_edge(input_node, node)
        return node
