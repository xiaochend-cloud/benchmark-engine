#!/usr/bin/env python3
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
"""Textual TUI for monitoring CopperDroid malware-detection pipeline progress.

Run this in a **separate terminal** while the pipeline is executing.

Usage
-----
    python3 copperdroid_monitor.py [--progress_file PATH]

The default progress file path is ``/tmp/copperdroid_progress.json`` (or
whatever ``COPPERDROID_PROGRESS_FILE`` is set to in the environment).

Install dependency
------------------
    pip install textual

Layout (2 × 2 grid)
-------------------
    ┌────────────────────────┬────────────────────────┐
    │  SOURCE & PREPROC      │  GRAPH CONSTRUCTION    │
    ├────────────────────────┼────────────────────────┤
    │  MLP TRAINING          │  GNN TRAINING          │
    └────────────────────────┴────────────────────────┘
"""

import json
import os
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import click

try:
    from textual.app import App
    from textual.app import ComposeResult
    from textual.containers import Grid
    from textual.widgets import Footer
    from textual.widgets import Header
    from textual.widgets import Static
except ImportError:
    print("ERROR: 'textual' is not installed.  Run:  pip install textual", file=sys.stderr)
    sys.exit(1)

REFRESH_INTERVAL = 1.0  # seconds between JSON polls


# ---------------------------------------------------------------------------
# Rich markup helpers
# ---------------------------------------------------------------------------

def _icon(status: str) -> str:
    return {
        "completed": "[bold green]✓[/bold green]",
        "in_progress": "[bold yellow]⟳[/bold yellow]",
        "error": "[bold red]✗[/bold red]",
        "waiting": "[dim]…[/dim]",
    }.get(status or "waiting", "[dim]—[/dim]")


def _bar(ratio: float, width: int = 22) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = int(ratio * width)
    empty = width - filled
    return "[green]" + "█" * filled + "[/green][dim]" + "░" * empty + "[/dim]"


def _fmt_metric(label: str, value) -> str:
    if value is None:
        return ""
    return f"  {label:<12} {value:.4f}\n"


# ---------------------------------------------------------------------------
# Panel widgets
# ---------------------------------------------------------------------------

class StagePanel(Static):
    DEFAULT_CSS = """
    StagePanel {
        border: round $accent;
        padding: 1 2;
        height: 100%;
        overflow-y: auto;
    }
    """


# ---------------------------------------------------------------------------
# Main TUI application
# ---------------------------------------------------------------------------

class PipelineMonitor(App):
    """4-panel real-time monitor for the CopperDroid pipeline."""

    CSS = """
    Screen {
        background: $surface;
    }

    Grid {
        grid-size: 2 2;
        grid-gutter: 1 2;
        margin: 1 2;
    }
    """

    TITLE = "CopperDroid Pipeline Monitor"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, progress_file: str):
        super().__init__()
        self._progress_file = progress_file
        self._data: dict = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Grid(
            StagePanel(id="panel_source"),
            StagePanel(id="panel_graph"),
            StagePanel(id="panel_mlp"),
            StagePanel(id="panel_gnn"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(REFRESH_INTERVAL, self._poll_and_render)

    # ------------------------------------------------------------------
    # Data polling
    # ------------------------------------------------------------------

    def _poll_and_render(self) -> None:
        try:
            self._data = json.loads(Path(self._progress_file).read_text())
        except FileNotFoundError:
            self._data = {}
            self.sub_title = f"Waiting for {self._progress_file} …"
            return
        except json.JSONDecodeError:
            return  # partial write in progress; keep stale data

        pipeline = self._data.get("pipeline", "—")
        started = self._data.get("started_at", "")
        if started:
            try:
                dt = datetime.fromisoformat(started)
                elapsed = datetime.now(tz=timezone.utc) - dt
                mins, secs = divmod(int(elapsed.total_seconds()), 60)
                self.sub_title = f"{pipeline}  |  elapsed {mins:02d}:{secs:02d}"
            except ValueError:
                self.sub_title = pipeline
        else:
            self.sub_title = pipeline

        stages = self._data.get("stages", {})
        self.query_one("#panel_source").update(self._render_source(stages))
        self.query_one("#panel_graph").update(self._render_graph(stages))
        self.query_one("#panel_mlp").update(self._render_training("mlp", "MLP TRAINING", stages))
        self.query_one("#panel_gnn").update(self._render_training("gnn", "GNN TRAINING", stages))

    # ------------------------------------------------------------------
    # Panel renderers
    # ------------------------------------------------------------------

    def _render_source(self, stages: dict) -> str:
        src = stages.get("source", {})
        pre = stages.get("preprocessing", {})

        lines = "[bold]SOURCE & PREPROCESSING[/bold]\n\n"

        # Source
        lines += f"Source   {_icon(src.get('status'))}  "
        lines += f"{src.get('message', '[dim]waiting…[/dim]')}\n"

        # Preprocessing
        if pre:
            lines += f"Preproc  {_icon(pre.get('status'))}\n"
            n_tr = pre.get("n_train")
            n_va = pre.get("n_val")
            n_te = pre.get("n_test")
            if n_tr is not None:
                lines += f"  Train / Val / Test  {n_tr} / {n_va} / {n_te}\n"
            if pre.get("split_seed") is not None:
                lines += f"  Seed  {pre['split_seed']}\n"
            if pre.get("num_features") is not None:
                lines += f"  Features  {pre['num_features']}\n"
        else:
            lines += "Preproc  [dim]waiting…[/dim]\n"

        return lines

    def _render_graph(self, stages: dict) -> str:
        g = stages.get("graph", {})

        lines = "[bold]GRAPH CONSTRUCTION[/bold]\n\n"

        if not g:
            lines += "[dim]Not active in this run[/dim]\n"
            lines += "\n[dim](GNN-only or parallel pipeline)[/dim]\n"
            return lines

        lines += f"Status  {_icon(g.get('status'))}\n"
        if g.get("n_nodes") is not None:
            lines += f"Nodes   {g['n_nodes']}\n"
        if g.get("k") is not None:
            lines += f"k       {g['k']}\n"
        if g.get("message"):
            lines += f"\n{g['message']}\n"

        return lines

    def _render_training(self, key: str, title: str, stages: dict) -> str:
        s = stages.get(key, {})

        lines = f"[bold]{title}[/bold]\n\n"

        if not s:
            lines += "[dim]Waiting…[/dim]\n"
            return lines

        status = s.get("status", "")
        lines += f"Status  {_icon(status)}\n"

        total = s.get("total_epochs")
        epoch = s.get("epoch", 0)

        if total:
            ratio = epoch / total
            lines += f"Epoch   {epoch} / {total}\n"
            lines += _bar(ratio) + f"  {ratio:.0%}\n"

        val_loss = s.get("val_loss")
        val_acc = s.get("val_acc")
        if val_loss is not None or val_acc is not None:
            lines += "\n"
            lines += _fmt_metric("Val loss", val_loss)
            lines += _fmt_metric("Val acc", val_acc)

        if status == "completed" and s.get("metrics"):
            m = s["metrics"]
            lines += "\n[bold]Test results[/bold]\n"
            lines += _fmt_metric("Accuracy", m.get("accuracy"))
            lines += _fmt_metric("F1", m.get("f1"))
            lines += _fmt_metric("Precision", m.get("precision"))
            lines += _fmt_metric("Recall", m.get("recall"))
            lines += _fmt_metric("Specificity", m.get("specificity"))
            lines += _fmt_metric("FPR", m.get("fpr"))
            lines += _fmt_metric("FNR", m.get("fnr"))

        return lines


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--progress_file",
    type=str,
    default=os.environ.get("COPPERDROID_PROGRESS_FILE", "/tmp/copperdroid_progress.json"),
    show_default=True,
    help="Path to the JSON progress file written by the pipeline.",
)
def main(progress_file: str):
    """Live 4-panel TUI for the CopperDroid pipeline.  Press [q] to quit."""
    app = PipelineMonitor(progress_file=progress_file)
    app.run()


if __name__ == "__main__":
    main()
