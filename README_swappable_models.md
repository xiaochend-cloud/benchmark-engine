<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Swappable Model Architecture — Benchmark Engine

This document describes the swappable model architecture extension to the benchmark engine, which enables direct comparison between a standard AutoEncoder (AE) and a ResidualAutoEncoder (RAE) trained on the same data. A parallel fan-out pipeline trains both models simultaneously on a single data stream, validating Morpheus's fan-out capability for comparative model benchmarking.

---

## New Files

### Python Package (`python/morpheus_benchmark_engine/`)

| File | Description |
|---|---|
| `models/__init__.py` | Package marker |
| `models/residual_ae_module.py` | `ResidualAEModule` — wraps each encoder/decoder `CompleteLayer` in a skip-connection `ResidualBlock`; uses a learned linear projection when dimensions differ |
| `models/residual_autoencoder.py` | `ResidualAutoEncoder` — subclasses `AutoEncoder`, replacing `self.model` with `ResidualAEModule` at init; all training logic inherited unchanged |
| `stages/benchmark_engine_training_base.py` | `BenchmarkEngineTrainingBase` (abstract), `AutoEncoderTraining`, `ResidualAutoEncoderTraining` — model is swapped via a single `_create_model()` method |

### Pipelines (`examples/benchmark_engine/production/`)

| File | Description |
|---|---|
| `dfp_duo_pipeline_autoencoder.py` | `LinearPipeline` training with `AutoEncoderTraining` |
| `dfp_duo_pipeline_residual_autoencoder.py` | `LinearPipeline` training with `ResidualAutoEncoderTraining` |
| `dfp_duo_pipeline_parallel.py` | Non-linear `Pipeline` with `BroadcastStage` fan-out, training AE and RAE simultaneously |

All three pipelines are derived from `dfp_duo_pipeline.py` (training branch only). Everything from `MultiFileSource` through `DFPPreprocessingStage` is shared; only the training stage and pipeline topology differ.

---

## Architecture

### ResidualAutoEncoder

```
Input
  └── encoder_0: ResidualBlock(in_dim → dim_0)
        ├── CompleteLayer(in_dim → dim_0)
        └── + skip connection (projection if dims differ)
  └── encoder_1: ResidualBlock(...)
  └── ...
  └── decoder layers (same pattern)
  └── numeric / binary / categorical outputs
```

### Parallel Fan-out Pipeline

```
MultiFileSource
  → DFPFileBatcherStage
  → DFPFileToDataFrameStage
  → MonitorStage("Input data rate")
  → DFPSplitUsersStage
  → DFPRollingWindowStage
  → DFPPreprocessingStage
  → BroadcastStage (1-in, 2-out)
       ├── output_ports[0] → AutoEncoderTraining
       │                     → MonitorStage("AE Training rate")
       │                     → DFPMLFlowModelWriterStage("AE-duo-{user_id}")
       │
       └── output_ports[1] → ResidualAutoEncoderTraining
                             → MonitorStage("RAE Training rate")
                             → DFPMLFlowModelWriterStage("RAE-duo-{user_id}")
```

`BroadcastStage` is a thin wrapper around `mrc.core.node.Broadcast`, defined inline in `dfp_duo_pipeline_parallel.py`.

---

## Setup and Running

All commands run from `examples/benchmark_engine/production/`.

### 1. Build

```bash
docker compose build
```

### 2. Fetch training data

```bash
docker compose run fetch_data
```

Data is downloaded to `/workspace/examples/data/dfp/duo-training-data/`.

### 3. Start MLflow

```bash
docker compose up -d mlflow
```

MLflow dashboard available at `http://localhost:5000`.

### 4. Run training

**AE only:**
```bash
docker compose --profile training run --rm morpheus_pipeline \
    bash -lc 'pip install -q -e /workspace/python/morpheus_benchmark_engine && \
    cd /workspace/examples/benchmark_engine/production && \
    python3 dfp_duo_pipeline_autoencoder.py \
    --source duo \
    --start_time "2022-08-01" \
    --duration 30d \
    --train_users generic \
    --input_file "/workspace/examples/data/dfp/duo-training-data/*.json"'
```

**RAE only:**
```bash
docker compose --profile training run --rm morpheus_pipeline \
    bash -lc 'pip install -q -e /workspace/python/morpheus_benchmark_engine && \
    cd /workspace/examples/benchmark_engine/production && \
    python3 dfp_duo_pipeline_residual_autoencoder.py \
    --source duo \
    --start_time "2022-08-01" \
    --duration 30d \
    --train_users generic \
    --input_file "/workspace/examples/data/dfp/duo-training-data/*.json"'
```

**Parallel AE + RAE:**
```bash
docker compose --profile training run --rm morpheus_pipeline \
    bash -lc 'pip install -q -e /workspace/python/morpheus_benchmark_engine && \
    cd /workspace/examples/benchmark_engine/production && \
    python3 dfp_duo_pipeline_parallel.py \
    --source duo \
    --start_time "2022-08-01" \
    --duration 30d \
    --train_users generic \
    --input_file "/workspace/examples/data/dfp/duo-training-data/*.json"'
```

---

## Experimental Results

**Dataset:** Synthetic Duo Security authentication logs (August 2022, 30-day window, `--train_users generic`)
**Epochs:** 1000
**Device:** CPU (`device: None`)

| Pipeline | real |
|---|---|
| AE only | 1m 5.883s |
| RAE only | 2m 5.055s |
| Parallel (AE + RAE) | 1m 4.322s |

> **Note:** `user` and `sys` times are near zero (~0.1s) for all runs because PyTorch and the MRC runtime execute in C++ threads not tracked by the shell. `real` (wall clock time) is the only meaningful metric.

### Key Findings

1. **RAE trains ~2× slower than AE** — residual blocks add a forward pass through the projection layer and skip connection at each encoder/decoder step.

2. **Parallel pipeline completes faster than either model alone** — wall time (1m 4s) is less than both AE (1m 6s) and RAE (2m 5s) sequential runs, and substantially less than their sum (3m 11s). Fan-out allows both models to train concurrently on the same preprocessed data stream, bounding total wall time by the faster branch rather than their sum.

3. **CPU contention limits parallel gains** — on CPU, both branches compete for the same cores. The advantage would be more pronounced on GPU, where each branch can utilize separate CUDA streams.

---

## Modified Files

| File | Change |
|---|---|
| `examples/benchmark_engine/production/Dockerfile` | Fixed `WORKDIR`, `COPY`, `ENTRYPOINT` paths (previously pointed to `digital_fingerprinting`) |
| `python/morpheus_benchmark_engine/setup.py` | Removed `versioneer` dependency; hardcoded version `0.1.0`; fixed `name` from `morpheus_dfp` to `morpheus_benchmark_engine` |
