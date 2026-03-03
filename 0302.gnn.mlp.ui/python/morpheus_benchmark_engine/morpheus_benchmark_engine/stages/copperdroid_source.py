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
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema
from morpheus_benchmark_engine.utils import progress_tracker

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

    def __init__(self, c: Config, csv_path: str, label_column: str = "Class"):
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
        n_features = len(df.columns) - 1

        # Log label distribution so unexpected values are immediately visible
        label_counts = df[self._label_column].value_counts().to_dict()
        logger.info(
            "Loaded %d samples, %d features. Label '%s' distribution: %s",
            n_samples, n_features, self._label_column, label_counts,
        )

        progress_tracker.update(
            "source",
            status="completed",
            message=f"Loaded {n_samples} samples, {n_features} features",
        )

        # Store the raw pandas DataFrame directly in metadata to avoid the
        # MessageMeta → cuDF → .to_pandas() round-trip that causes
        # sklearn/PyTorch incompatibility.  An empty payload satisfies Morpheus.
        msg = ControlMessage()
        msg.payload(MessageMeta(pd.DataFrame()))
        msg.set_metadata("raw_df", df)
        msg.set_metadata("label_column", self._label_column)

        yield msg

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        node = builder.make_source(self.unique_name, self._generate_frames())
        return node
