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
"""Parallel MLP + GNN malware detection pipeline for the CopperDroid dataset.

Fan-out architecture: a single preprocessing stream is broadcast to two
independent training branches.  The MLP branch trains directly on tabular
features; the GNN branch first constructs a k-NN graph.

Pipeline structure:
  CopperDroidSource
    → MonitorStage("Load rate")
    → MalwarePreprocessingStage
    → BroadcastStage (1-in, 2-out)
         ├── output_ports[0] → MLPTrainingStage
         │                     → MonitorStage("MLP training rate")
         │                     → MalwareMLFlowWriterStage
         │
         └── output_ports[1] → GraphConstructionStage
                               → GNNTrainingStage
                               → MonitorStage("GNN training rate")
                               → MalwareMLFlowWriterStage
"""

import logging
import os

import click
import mlflow
import mrc
import mrc.core.operators as ops
from mrc.core.node import Broadcast

from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import parse_log_level
from morpheus.config import Config
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pipeline import Pipeline
from morpheus.pipeline.stage import Stage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.utils.logger import configure_logging

from morpheus_benchmark_engine.stages.copperdroid_source import CopperDroidSource
from morpheus_benchmark_engine.stages.graph_construction_stage import GraphConstructionStage
from morpheus_benchmark_engine.stages.malware_mlflow_writer import MalwareMLFlowWriterStage
from morpheus_benchmark_engine.stages.malware_preprocessing_stage import MalwarePreprocessingStage
from morpheus_benchmark_engine.stages.malware_training_base import GNNTrainingStage
from morpheus_benchmark_engine.stages.malware_training_base import MLPTrainingStage
from morpheus_benchmark_engine.utils import progress_tracker


# ---------------------------------------------------------------------------
# BroadcastStage — reused from dfp_duo_pipeline_parallel.py
# ---------------------------------------------------------------------------

class BroadcastStage(GpuAndCpuMixin, Stage):
    """Broadcasts a single input port to *num_outputs* identical output ports."""

    def __init__(self, c: Config, num_outputs: int = 2):
        super().__init__(c)
        self._num_outputs = num_outputs
        self._create_ports(1, num_outputs)

    @property
    def name(self) -> str:
        return "broadcast"

    def supports_cpp_node(self):
        return False

    def compute_schema(self, schema: StageSchema):
        for port_schema in schema.output_schemas:
            port_schema.set_type(schema.input_type)

    def _build(self, builder: mrc.Builder, input_nodes: list) -> list:
        broadcast = Broadcast(builder, "broadcast")
        builder.make_edge(input_nodes[0], broadcast)

        output_nodes = []
        for i in range(self._num_outputs):
            node = builder.make_node(f"pass_{i}", ops.map(lambda x: x))
            builder.make_edge(broadcast, node)
            output_nodes.append(node)

        return output_nodes


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--csv_path",
    type=str,
    default="/workspace/examples/data/copperdroid/feature_vectors_top_combined.csv",
    show_default=True,
    help="Path to the CopperDroid feature-vector CSV.",
)
@click.option("--label_column", type=str, default="label", show_default=True,
              help="Name of the label column in the CSV.")
@click.option("--val_size", type=float, default=0.10, show_default=True,
              help="Fraction of data for validation.")
@click.option("--test_size", type=float, default=0.10, show_default=True,
              help="Fraction of data for testing.")
@click.option("--split_seed", type=int, default=42, show_default=True,
              help="Random seed for the train/val/test split.")
# MLP options
@click.option("--mlp_hidden_dims", type=str, default="256,128", show_default=True,
              help="Comma-separated hidden layer sizes for the MLP.")
@click.option("--mlp_dropout", type=float, default=0.3, show_default=True,
              help="MLP dropout probability.")
@click.option("--mlp_epochs", type=int, default=100, show_default=True,
              help="MLP training epochs.")
@click.option("--mlp_batch_size", type=int, default=256, show_default=True,
              help="MLP mini-batch size.")
# GNN options
@click.option("--k", type=int, default=10, show_default=True,
              help="Number of nearest neighbours for the k-NN graph.")
@click.option("--gnn_hidden_dim", type=int, default=128, show_default=True,
              help="GCN hidden dimension.")
@click.option("--gnn_dropout", type=float, default=0.5, show_default=True,
              help="GNN dropout probability.")
@click.option("--gnn_epochs", type=int, default=200, show_default=True,
              help="GNN training epochs.")
# Shared options
@click.option("--lr", type=float, default=1e-3, show_default=True, help="Adam learning rate.")
@click.option("--weight_seed", type=int, default=0, show_default=True,
              help="Seed for model weight initialisation.")
@click.option("--device", type=str, default=None, show_default=True,
              help="'cuda' or 'cpu'. Auto-detects if not set.")
@click.option("--experiment_name", type=str, default="copperdroid/malware", show_default=True,
              help="MLflow experiment name.")
@click.option("--tracking_uri", type=str, default="http://mlflow:5000", show_default=True,
              help="MLflow tracking server URI.")
@click.option(
    "--log_level",
    default="INFO",
    type=click.Choice(get_log_levels(), case_sensitive=False),
    callback=parse_log_level,
    help="Logging level.",
)
def run_pipeline(
    csv_path, label_column, val_size, test_size, split_seed,
    mlp_hidden_dims, mlp_dropout, mlp_epochs, mlp_batch_size,
    k, gnn_hidden_dim, gnn_dropout, gnn_epochs,
    lr, weight_seed, device,
    experiment_name, tracking_uri, log_level,
):
    """Parallel MLP + GNN malware classification pipeline for the CopperDroid dataset."""
    configure_logging(log_level=log_level)
    logging.getLogger("mlflow").setLevel(log_level)

    mlflow.set_tracking_uri(tracking_uri)
    progress_tracker.init("copperdroid-parallel")

    hidden = tuple(int(x) for x in mlp_hidden_dims.split(","))

    config = Config()
    config.num_threads = len(os.sched_getaffinity(0))

    pipeline = Pipeline(config)

    # ------------------------------------------------------------------
    # Shared preprocessing stages
    # ------------------------------------------------------------------
    src = pipeline.add_stage(
        CopperDroidSource(config, csv_path=csv_path, label_column=label_column)
    )

    mon_in = pipeline.add_stage(MonitorStage(config, description="Load rate"))

    preproc = pipeline.add_stage(
        MalwarePreprocessingStage(
            config,
            val_size=val_size,
            test_size=test_size,
            split_seed=split_seed,
            label_column=label_column,
        )
    )

    bcast = pipeline.add_stage(BroadcastStage(config, num_outputs=2))

    # ------------------------------------------------------------------
    # MLP branch (output_ports[0])
    # ------------------------------------------------------------------
    mlp_trn = pipeline.add_stage(
        MLPTrainingStage(
            config,
            hidden_dims=hidden,
            dropout=mlp_dropout,
            epochs=mlp_epochs,
            lr=lr,
            batch_size=mlp_batch_size,
            weight_seed=weight_seed,
            device=device,
        )
    )

    mon_mlp = pipeline.add_stage(MonitorStage(config, description="MLP training rate", smoothing=0.001))

    mlf_mlp = pipeline.add_stage(
        MalwareMLFlowWriterStage(
            config,
            experiment_name=experiment_name,
            model_name_formatter="copperdroid-{arch}",
            tracking_uri=tracking_uri,
        )
    )

    # ------------------------------------------------------------------
    # GNN branch (output_ports[1])
    # ------------------------------------------------------------------
    graph = pipeline.add_stage(GraphConstructionStage(config, k=k))

    gnn_trn = pipeline.add_stage(
        GNNTrainingStage(
            config,
            hidden_dim=gnn_hidden_dim,
            dropout=gnn_dropout,
            epochs=gnn_epochs,
            lr=lr,
            weight_seed=weight_seed,
            device=device,
        )
    )

    mon_gnn = pipeline.add_stage(MonitorStage(config, description="GNN training rate", smoothing=0.001))

    mlf_gnn = pipeline.add_stage(
        MalwareMLFlowWriterStage(
            config,
            experiment_name=experiment_name,
            model_name_formatter="copperdroid-{arch}",
            tracking_uri=tracking_uri,
        )
    )

    # ------------------------------------------------------------------
    # Wire edges
    # ------------------------------------------------------------------
    # Linear chain up to broadcast
    pipeline.add_edge(src, mon_in)
    pipeline.add_edge(mon_in, preproc)
    pipeline.add_edge(preproc, bcast)

    # Fan-out: connect each broadcast output port to its branch
    pipeline.add_edge(bcast.output_ports[0], mlp_trn)
    pipeline.add_edge(bcast.output_ports[1], graph)

    # MLP downstream chain
    pipeline.add_edge(mlp_trn, mon_mlp)
    pipeline.add_edge(mon_mlp, mlf_mlp)

    # GNN downstream chain
    pipeline.add_edge(graph, gnn_trn)
    pipeline.add_edge(gnn_trn, mon_gnn)
    pipeline.add_edge(mon_gnn, mlf_gnn)

    pipeline.run()


if __name__ == "__main__":
    run_pipeline(obj={}, auto_envvar_prefix="COPPERDROID", show_default=True,
                 prog_name="copperdroid-parallel")
