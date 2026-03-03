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
"""Source stage that loads a CopperDroid feature-vector CSV into the pipeline."""

import logging
import typing

import mrc
import pandas as pd

from morpheus.config import Config
from morpheus_benchmark_engine.utils import progress_tracker
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(f"morpheus.{__name__}")


class CopperDroidSource(SingleOutputSource):
    """Read a CopperDroid feature-vector CSV and emit a single ControlMessage.

    The CSV is expected to have one row per APK and one column per feature,
    plus a label column (default: ``"label"``).  All columns except the label
    are treated as numeric features.

    Parameters
    ----------
    c : Config
        Pipeline configuration instance.
    csv_path : str
        Path to the CopperDroid feature-vector CSV file.
    label_column : str
        Name of the label column in the CSV.  Default ``"label"``.
    """

    def __init__(self, c: Config, csv_path: str, label_column: str = "label"):
        super().__init__(c)
        self._csv_path = csv_path
        self._label_column = label_column

    @property
    def name(self) -> str:
        return "copperdroid-source"

    @property
    def input_count(self) -> typing.Optional[int]:
        return 1

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def supports_cpp_node(self):
        return False

    def _generate_frames(self) -> typing.Iterable[ControlMessage]:
        logger.info("Loading CopperDroid dataset from '%s'", self._csv_path)
        df = pd.read_csv(self._csv_path)

        if self._label_column not in df.columns:
            raise ValueError(
                f"Label column '{self._label_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

        n_samples = len(df)
        n_cols = len(df.columns)
        logger.info("Loaded %d samples, %d columns (including label)", n_samples, n_cols)
        progress_tracker.update(
            "source",
            status="completed",
            message=f"Loaded {n_samples} samples, {n_cols - 1} features",
        )

        meta = MessageMeta(df)
        msg = ControlMessage()
        msg.payload(meta)
        msg.set_metadata("label_column", self._label_column)
        msg.set_metadata("csv_path", self._csv_path)

        yield msg

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        node = builder.make_source(self.unique_name, self._generate_frames())
        return node
