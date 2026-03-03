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
"""Lightweight progress-tracking helper for the CopperDroid pipeline.

Stages call ``update()`` at key milestones; the Textual TUI reads the
resulting JSON file to render live progress panels.

Configuration
-------------
Set the environment variable ``COPPERDROID_PROGRESS_FILE`` to the path
where the JSON file should be written.  Defaults to
``/tmp/copperdroid_progress.json``.

If the env var is not set AND the default path is not writable the
functions are no-ops — the pipeline continues unaffected.

Thread safety
-------------
Multiple stages run in separate MRC threads and may call ``update()``
concurrently.  Writes are serialised with a process-level lock and use
an atomic ``os.replace()`` so the reader never sees a partial file.
"""

import json
import logging
import os
import threading
from datetime import datetime
from datetime import timezone
from pathlib import Path

logger = logging.getLogger(f"morpheus.{__name__}")

PROGRESS_FILE_ENV = "COPPERDROID_PROGRESS_FILE"
DEFAULT_PATH = "/tmp/copperdroid_progress.json"

_lock = threading.Lock()


def _get_path() -> str:
    return os.environ.get(PROGRESS_FILE_ENV, DEFAULT_PATH)


def init(pipeline_name: str) -> None:
    """Create (or reset) the progress file for a new pipeline run.

    Call this once at the start of a pipeline script before any stages run.

    Parameters
    ----------
    pipeline_name : str
        Human-readable label shown in the TUI header.
    """
    path = _get_path()
    data = {
        "pipeline": pipeline_name,
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
        "stages": {},
    }
    try:
        _write(path, data)
    except OSError as exc:
        logger.debug("progress_tracker.init: could not write '%s': %s", path, exc)


def update(stage: str, **kwargs) -> None:
    """Write or update a stage entry in the progress file.

    Parameters
    ----------
    stage : str
        Stage key (e.g. ``"source"``, ``"mlp"``, ``"gnn"``).
    **kwargs
        Arbitrary key/value pairs recorded under this stage.
        Common keys: ``status`` (``"in_progress"`` | ``"completed"`` | ``"error"``),
        ``epoch``, ``total_epochs``, ``val_loss``, ``val_acc``, ``metrics``.
    """
    path = _get_path()
    with _lock:
        try:
            raw = Path(path).read_text()
            data = json.loads(raw)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {
                "pipeline": "unknown",
                "started_at": datetime.now(tz=timezone.utc).isoformat(),
                "stages": {},
            }

        # Merge kwargs into the stage entry (preserving previous keys)
        entry = data["stages"].get(stage, {})
        entry.update(kwargs)
        entry["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
        data["stages"][stage] = entry

        try:
            _write(path, data)
        except OSError as exc:
            logger.debug("progress_tracker.update: could not write '%s': %s", path, exc)


def _write(path: str, data: dict) -> None:
    """Atomically write *data* as JSON to *path*."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(tmp, path)  # atomic on POSIX; best-effort on Windows
