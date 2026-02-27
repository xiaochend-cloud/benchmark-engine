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
"""Parallel AE + RAE training pipeline for Duo Authentication logs.

Fan-out architecture: a single preprocessing stream is broadcast to two
independent training branches (AutoEncoder and ResidualAutoEncoder) that each
write their results to MLflow.
"""

import functools
import logging
import os
import typing
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import click
import mlflow
import mrc
import mrc.core.operators as ops
import pandas as pd
from mrc.core.node import Broadcast

from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import get_package_relative_file
from morpheus.cli.utils import parse_log_level
from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pipeline import Pipeline
from morpheus.pipeline.stage import Stage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.utils.column_info import BoolColumn
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import DateTimeColumn
from morpheus.utils.column_info import DistinctIncrementColumn
from morpheus.utils.column_info import IncrementColumn
from morpheus.utils.column_info import RenameColumn
from morpheus.utils.column_info import StringCatColumn
from morpheus.utils.file_utils import date_extractor
from morpheus.utils.file_utils import load_labels_file
from morpheus.utils.logger import configure_logging
from morpheus_benchmark_engine.stages.benchmark_engine_training_base import AutoEncoderTraining
from morpheus_benchmark_engine.stages.benchmark_engine_training_base import ResidualAutoEncoderTraining
from morpheus_benchmark_engine.stages.dfp_file_batcher_stage import DFPFileBatcherStage
from morpheus_benchmark_engine.stages.dfp_file_to_df import DFPFileToDataFrameStage
from morpheus_benchmark_engine.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
from morpheus_benchmark_engine.stages.dfp_preprocessing_stage import DFPPreprocessingStage
from morpheus_benchmark_engine.stages.dfp_rolling_window_stage import DFPRollingWindowStage
from morpheus_benchmark_engine.stages.dfp_split_users_stage import DFPSplitUsersStage
from morpheus_benchmark_engine.stages.multi_file_source import MultiFileSource
from morpheus_benchmark_engine.utils.regex_utils import iso_date_regex


# ---------------------------------------------------------------------------
# BroadcastStage — 1-input, N-output fan-out stage
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
# Helpers
# ---------------------------------------------------------------------------

def _file_type_name_to_enum(file_type: str) -> FileTypes:
    if file_type == "JSON":
        return FileTypes.JSON
    if file_type == "CSV":
        return FileTypes.CSV
    if file_type == "PARQUET":
        return FileTypes.PARQUET
    return FileTypes.Auto


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--source",
    type=click.Choice(["duo", "azure"], case_sensitive=False),
    required=True,
    help="Indicates what type of logs are going to be used in the workload.",
)
@click.option(
    "--train_users",
    type=click.Choice(["all", "generic", "individual"], case_sensitive=False),
    default="generic",
    help="Indicates whether to train per user or a generic model for all users.",
)
@click.option("--skip_user", multiple=True, type=str, help="User IDs to skip.")
@click.option("--only_user", multiple=True, type=str, help="Only include these user IDs.")
@click.option(
    "--start_time",
    type=click.DateTime(
        formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z",
                 "%Y-%m-%d %H:%M:%S%z"]),
    default=None,
    help="Start of the time window.",
)
@click.option("--duration", type=str, default="60d", help="Duration from start_time.")
@click.option("--cache_dir", type=str, default="./.cache/dfp", show_envvar=True, help="Cache directory.")
@click.option("--log_level",
              default="INFO",
              type=click.Choice(get_log_levels(), case_sensitive=False),
              callback=parse_log_level,
              help="Logging level.")
@click.option("--sample_rate_s", type=int, default=0, show_envvar=True, help="Sample rate in seconds.")
@click.option(
    "--input_file",
    "-f",
    type=str,
    multiple=True,
    help="List of input files or glob patterns.",
)
@click.option("--file_type_override",
              "-t",
              type=click.Choice(["AUTO", "JSON", "CSV", "PARQUET"], case_sensitive=False),
              default="JSON",
              callback=lambda _, __, value: None if value is None else _file_type_name_to_enum(value))
@click.option("--watch_inputs", type=bool, is_flag=True, default=False)
@click.option("--watch_interval", type=float, default=1.0)
@click.option("--tracking_uri", type=str, default="http://mlflow:5000", help="MLflow tracking URI.")
def run_pipeline(source,
                 train_users,
                 skip_user: typing.Tuple[str],
                 only_user: typing.Tuple[str],
                 start_time: datetime,
                 duration,
                 cache_dir,
                 log_level,
                 sample_rate_s,
                 file_type_override,
                 **kwargs):
    """Runs the parallel AE + RAE training pipeline with fan-out."""
    include_generic = train_users in ("all", "generic")
    include_individual = train_users != "generic"

    skip_users = list(skip_user)
    only_users = list(only_user)

    duration = timedelta(seconds=pd.Timedelta(duration).total_seconds())
    if start_time is None:
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - duration
    else:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        end_time = start_time + duration

    configure_logging(log_level=log_level)
    logging.getLogger("mlflow").setLevel(log_level)

    if len(skip_users) > 0 and len(only_users) > 0:
        logging.error("--skip_user and --only_user are mutually exclusive.")

    logger = logging.getLogger("morpheus.{__name__}")
    logger.info("Running parallel AE + RAE training pipeline")
    logger.info("Train generic_user: %s", include_generic)
    logger.info("Start Time: %s", start_time)
    logger.info("Duration: %s", duration)

    if "tracking_uri" in kwargs:
        mlflow.set_tracking_uri(kwargs["tracking_uri"])
        logger.info("Tracking URI: %s", mlflow.get_tracking_uri())

    config = Config()
    config.num_threads = len(os.sched_getaffinity(0))
    config.ae = ConfigAutoEncoder()
    config.ae.feature_columns = load_labels_file(get_package_relative_file("data/columns_ae_duo.txt"))
    config.ae.userid_column_name = "username"
    config.ae.timestamp_column_name = "timestamp"

    source_column_info = [
        DateTimeColumn(name=config.ae.timestamp_column_name, dtype=datetime, input_name="timestamp"),
        RenameColumn(name=config.ae.userid_column_name, dtype=str, input_name="user.name"),
        RenameColumn(name="accessdevicebrowser", dtype=str, input_name="access_device.browser"),
        RenameColumn(name="accessdeviceos", dtype=str, input_name="access_device.os"),
        StringCatColumn(name="location",
                        dtype=str,
                        input_columns=[
                            "access_device.location.city",
                            "access_device.location.state",
                            "access_device.location.country"
                        ],
                        sep=", "),
        RenameColumn(name="authdevicename", dtype=str, input_name="auth_device.name"),
        BoolColumn(name="result",
                   dtype=bool,
                   input_name="result",
                   true_values=["success", "SUCCESS"],
                   false_values=["denied", "DENIED", "FRAUD"]),
        ColumnInfo(name="reason", dtype=str),
    ]

    source_schema = DataFrameInputSchema(json_columns=["access_device", "application", "auth_device", "user"],
                                         column_info=source_column_info)

    preprocess_column_info = [
        ColumnInfo(name=config.ae.timestamp_column_name, dtype=datetime),
        ColumnInfo(name=config.ae.userid_column_name, dtype=str),
        ColumnInfo(name="accessdevicebrowser", dtype=str),
        ColumnInfo(name="accessdeviceos", dtype=str),
        ColumnInfo(name="authdevicename", dtype=str),
        ColumnInfo(name="result", dtype=bool),
        ColumnInfo(name="reason", dtype=str),
        IncrementColumn(name="logcount",
                        dtype=int,
                        input_name=config.ae.timestamp_column_name,
                        groupby_column=config.ae.userid_column_name),
        DistinctIncrementColumn(name="locincrement",
                                dtype=int,
                                input_name="location",
                                groupby_column=config.ae.userid_column_name,
                                timestamp_column=config.ae.timestamp_column_name),
    ]
    preprocess_schema = DataFrameInputSchema(column_info=preprocess_column_info, preserve_columns=["_batch_id"])

    # -----------------------------------------------------------------------
    # Build the non-linear Pipeline
    # -----------------------------------------------------------------------
    pipeline = Pipeline(config)

    src = pipeline.add_stage(
        MultiFileSource(config,
                        filenames=list(kwargs["input_file"]),
                        watch=kwargs["watch_inputs"],
                        watch_interval=kwargs["watch_interval"]))

    batcher = pipeline.add_stage(
        DFPFileBatcherStage(config,
                            period=None,
                            sampling=f"{sample_rate_s}S" if sample_rate_s > 0 else None,
                            date_conversion_func=functools.partial(date_extractor, filename_regex=iso_date_regex),
                            start_time=start_time,
                            end_time=end_time))

    parser_kwargs = None
    if file_type_override == FileTypes.JSON:
        parser_kwargs = {"lines": False, "orient": "records"}

    to_df = pipeline.add_stage(
        DFPFileToDataFrameStage(config,
                                schema=source_schema,
                                file_type=file_type_override,
                                parser_kwargs=parser_kwargs,
                                cache_dir=cache_dir))

    mon_in = pipeline.add_stage(MonitorStage(config, description="Input data rate"))

    split = pipeline.add_stage(
        DFPSplitUsersStage(config,
                           include_generic=include_generic,
                           include_individual=include_individual,
                           skip_users=skip_users,
                           only_users=only_users))

    window = pipeline.add_stage(
        DFPRollingWindowStage(config, min_history=300, min_increment=300, max_history="60d", cache_dir=cache_dir))

    preproc = pipeline.add_stage(DFPPreprocessingStage(config, input_schema=preprocess_schema))

    bcast = pipeline.add_stage(BroadcastStage(config, num_outputs=2))

    ae_trn = pipeline.add_stage(AutoEncoderTraining(config, validation_size=0.10))
    rae_trn = pipeline.add_stage(ResidualAutoEncoderTraining(config, validation_size=0.10))

    mon_ae = pipeline.add_stage(MonitorStage(config, description="AE Training rate", smoothing=0.001))
    mon_rae = pipeline.add_stage(MonitorStage(config, description="RAE Training rate", smoothing=0.001))

    mlf_ae = pipeline.add_stage(
        DFPMLFlowModelWriterStage(config,
                                  model_name_formatter="AE-duo-{user_id}",
                                  experiment_name_formatter="benchmark/duo/ae/{reg_model_name}"))

    mlf_rae = pipeline.add_stage(
        DFPMLFlowModelWriterStage(config,
                                  model_name_formatter="RAE-duo-{user_id}",
                                  experiment_name_formatter="benchmark/duo/rae/{reg_model_name}"))

    # Linear chain up to broadcast
    pipeline.add_edge(src, batcher)
    pipeline.add_edge(batcher, to_df)
    pipeline.add_edge(to_df, mon_in)
    pipeline.add_edge(mon_in, split)
    pipeline.add_edge(split, window)
    pipeline.add_edge(window, preproc)
    pipeline.add_edge(preproc, bcast)

    # Fan-out: connect each broadcast output port to its training branch
    pipeline.add_edge(bcast.output_ports[0], ae_trn)
    pipeline.add_edge(bcast.output_ports[1], rae_trn)

    # Independent downstream chains
    pipeline.add_edge(ae_trn, mon_ae)
    pipeline.add_edge(mon_ae, mlf_ae)
    pipeline.add_edge(rae_trn, mon_rae)
    pipeline.add_edge(mon_rae, mlf_rae)

    pipeline.run()


if __name__ == "__main__":
    run_pipeline(obj={}, auto_envvar_prefix="DFP", show_default=True, prog_name="dfp")
