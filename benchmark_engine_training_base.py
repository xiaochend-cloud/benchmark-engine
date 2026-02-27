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
"""Swappable training stage base and concrete implementations (AE and Residual AE)."""
import logging
import typing
from abc import abstractmethod

import mrc
from mrc.core import operators as ops
from sklearn.model_selection import train_test_split

import cudf

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.models.dfencoder import AutoEncoder
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

from morpheus_benchmark_engine.models.residual_autoencoder import ResidualAutoEncoder

logger = logging.getLogger(f"morpheus.{__name__}")


class BenchmarkEngineTrainingBase(SinglePortStage):
    """Abstract base training stage that delegates model creation to subclasses.

    Parameters
    ----------
    c : Config
        Pipeline configuration instance.
    model_kwargs : dict, optional
        Keyword arguments forwarded to the model constructor.
    epochs : int
        Number of training epochs.
    validation_size : float
        Fraction of data to use for validation (must be in [0, 1)).
    """

    def __init__(self, c: Config, model_kwargs: dict = None, epochs: int = 30, validation_size: float = 0.0):
        super().__init__(c)

        self._model_kwargs = {
            "encoder_layers": [512, 500],
            "decoder_layers": [512],
            "activation": "relu",
            "swap_probability": 0.2,
            "learning_rate": 0.001,
            "learning_rate_decay": 0.99,
            "batch_size": 512,
            "verbose": False,
            "optimizer": "sgd",
            "scaler": "standard",
            "min_cats": 1,
            "progress_bar": False,
            "device": None,
            "patience": -1,
        }

        self._model_kwargs.update(model_kwargs if model_kwargs is not None else {})

        self._epochs = epochs

        if 0.0 <= validation_size < 1.0:
            self._validation_size = validation_size
        else:
            raise ValueError(f"validation_size={validation_size} must be in [0, 1)")

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage, )

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(schema.input_type)

    @abstractmethod
    def _create_model(self):
        """Return a freshly instantiated model object."""

    def on_data(self, message: ControlMessage) -> ControlMessage:
        """Train the model and attach it to the output message."""
        if message is None or message.payload().count == 0:
            return None

        user_id = message.get_metadata("user_id")

        model = self._create_model()

        train_df = message.payload().copy_dataframe()

        if isinstance(train_df, cudf.DataFrame):
            train_df = train_df.to_pandas()

        train_df = train_df[train_df.columns.intersection(self._config.ae.feature_columns)]
        validation_df = None
        run_validation = False

        if self._validation_size > 0.0:
            train_df, validation_df = train_test_split(train_df, test_size=self._validation_size, shuffle=False)
            run_validation = True

        logger.debug("Training %s model for user: '%s'...", self.name, user_id)
        model.fit(train_df, epochs=self._epochs, validation_data=validation_df, run_validation=run_validation)
        logger.debug("Training %s model for user: '%s'... Complete.", self.name, user_id)

        output_message = ControlMessage()
        output_message.payload(message.payload())
        output_message.set_metadata("user_id", user_id)
        output_message.set_metadata("model", model)

        return output_message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data), ops.filter(lambda x: x is not None))
        builder.make_edge(input_node, node)
        return node


class AutoEncoderTraining(BenchmarkEngineTrainingBase):
    """Training stage that uses the standard AutoEncoder."""

    @property
    def name(self) -> str:
        return "ae-training"

    def _create_model(self):
        return AutoEncoder(**self._model_kwargs)


class ResidualAutoEncoderTraining(BenchmarkEngineTrainingBase):
    """Training stage that uses the ResidualAutoEncoder (skip connections)."""

    @property
    def name(self) -> str:
        return "rae-training"

    def _create_model(self):
        return ResidualAutoEncoder(**self._model_kwargs)
