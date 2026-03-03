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
"""MLP-only malware detection pipeline for the CopperDroid dataset.

Pipeline structure:
  CopperDroidSource
    → MonitorStage("Load rate")
    → MalwarePreprocessingStage
    → MLPTrainingStage
    → MonitorStage("MLP training rate")
    → MalwareMLFlowWriterStage
"""

import logging
import os

import click
import mlflow

from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import parse_log_level
from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.utils.logger import configure_logging

from morpheus_benchmark_engine.stages.copperdroid_source import CopperDroidSource
from morpheus_benchmark_engine.utils import progress_tracker
from morpheus_benchmark_engine.stages.malware_mlflow_writer import MalwareMLFlowWriterStage
from morpheus_benchmark_engine.stages.malware_preprocessing_stage import MalwarePreprocessingStage
from morpheus_benchmark_engine.stages.malware_training_base import MLPTrainingStage


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
@click.option("--epochs", type=int, default=100, show_default=True,
              help="Number of training epochs.")
@click.option("--lr", type=float, default=1e-3, show_default=True, help="Adam learning rate.")
@click.option("--batch_size", type=int, default=256, show_default=True, help="Mini-batch size.")
@click.option("--hidden_dims", type=str, default="256,128", show_default=True,
              help="Comma-separated hidden layer sizes for the MLP.")
@click.option("--dropout", type=float, default=0.3, show_default=True, help="Dropout probability.")
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
    epochs, lr, batch_size, hidden_dims, dropout, weight_seed, device,
    experiment_name, tracking_uri, log_level,
):
    """MLP-only malware classification pipeline for the CopperDroid dataset."""
    configure_logging(log_level=log_level)
    logging.getLogger("mlflow").setLevel(log_level)

    mlflow.set_tracking_uri(tracking_uri)
    progress_tracker.init("copperdroid-mlp")

    hidden = tuple(int(x) for x in hidden_dims.split(","))

    config = Config()
    config.num_threads = len(os.sched_getaffinity(0))

    pipeline = LinearPipeline(config)

    pipeline.set_source(CopperDroidSource(config, csv_path=csv_path, label_column=label_column))

    pipeline.add_stage(MonitorStage(config, description="Load rate"))

    pipeline.add_stage(
        MalwarePreprocessingStage(
            config,
            val_size=val_size,
            test_size=test_size,
            split_seed=split_seed,
            label_column=label_column,
        )
    )

    pipeline.add_stage(
        MLPTrainingStage(
            config,
            hidden_dims=hidden,
            dropout=dropout,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            weight_seed=weight_seed,
            device=device,
        )
    )

    pipeline.add_stage(MonitorStage(config, description="MLP training rate", smoothing=0.001))

    pipeline.add_stage(
        MalwareMLFlowWriterStage(
            config,
            experiment_name=experiment_name,
            model_name_formatter="copperdroid-{arch}",
            tracking_uri=tracking_uri,
        )
    )

    pipeline.run()


if __name__ == "__main__":
    run_pipeline(obj={}, auto_envvar_prefix="COPPERDROID", show_default=True, prog_name="copperdroid-mlp")
