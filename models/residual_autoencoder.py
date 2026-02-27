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
"""ResidualAutoEncoder — high-level wrapper that uses ResidualAEModule."""

from morpheus.models.dfencoder.autoencoder import AutoEncoder

from morpheus_benchmark_engine.models.residual_ae_module import ResidualAEModule


class ResidualAutoEncoder(AutoEncoder):
    """AutoEncoder variant that uses ResidualAEModule for skip-connection layers.

    All training logic (`fit`, `_build_model`, `_train_for_epochs`, etc.) is
    inherited unchanged from `AutoEncoder`.  Only `self.model` is replaced with
    a `ResidualAEModule` instance at construction time.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Replace the plain AEModule with the residual variant, preserving all
        # constructor arguments that were forwarded to the original module.
        self.model = ResidualAEModule(
            verbose=self.verbose,
            encoder_layers=self.model.encoder_layers,
            decoder_layers=self.model.decoder_layers,
            encoder_dropout=self.model.encoder_dropout,
            decoder_dropout=self.model.decoder_dropout,
            encoder_activations=self.model.encoder_activations,
            decoder_activations=self.model.decoder_activations,
            activation=self.model.activation,
            device=self.device,
        )
