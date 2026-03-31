"""Shared Weights & Biases helpers for the local trainer scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def add_wandb_args(parser: argparse.ArgumentParser, *, default_project: str) -> None:
    """Add shared W&B CLI flags to a trainer parser.

    :param argparse.ArgumentParser parser: Parser to extend.
    :param str default_project: Default W&B project name.
    """

    parser.add_argument(
        "--wandb",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("WANDB", "0")),
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", default_project),
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", ""),
    )
    parser.add_argument(
        "--wandb-group",
        default=os.environ.get("WANDB_GROUP", ""),
    )
    parser.add_argument(
        "--wandb-run-name",
        default=os.environ.get("WANDB_RUN_NAME", ""),
    )
    parser.add_argument(
        "--wandb-tags",
        default=os.environ.get("WANDB_TAGS", ""),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument(
        "--wandb-watch-log",
        choices=("gradients", "all"),
        default=os.environ.get("WANDB_WATCH_LOG", "gradients"),
    )
    parser.add_argument(
        "--wandb-watch-log-freq",
        type=int,
        default=int(os.environ.get("WANDB_WATCH_LOG_FREQ", "100")),
    )


def parse_wandb_tags(raw_tags: str) -> list[str]:
    """Split a comma-separated W&B tag string into distinct tags.

    :param str raw_tags: Comma-separated tag string.
    :return list[str]: Normalized tag list.
    """

    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def build_wandb_config(
    args: object, extra_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a W&B config payload from parsed CLI args plus derived values.

    :param object args: Parsed trainer args object.
    :param dict[str, Any] | None extra_config: Additional derived config values.
    :return dict[str, Any]: Serializable config payload.
    """

    config = dict(vars(args))
    if extra_config:
        config.update(extra_config)
    return config


def maybe_init_wandb(
    args: object,
    *,
    master_process: bool,
    output_dir: Path,
    config: dict[str, Any],
) -> Any | None:
    """Initialize a W&B run when enabled on rank 0.

    :param object args: Parsed trainer args object.
    :param bool master_process: Whether this process owns external logging.
    :param Path output_dir: Per-run output directory.
    :param dict[str, Any] config: Config payload to attach to the run.
    :return Any: W&B run object or ``None`` when disabled.
    """

    if not master_process or not getattr(args, "wandb", False):
        return None

    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group or None,
        name=args.wandb_run_name or args.run_id,
        tags=parse_wandb_tags(args.wandb_tags),
        config=config,
        dir=str(output_dir),
        mode=args.wandb_mode,
    )


def maybe_watch_wandb(run: Any, model: object, *, log: str, log_freq: int) -> None:
    """Attach W&B gradient/parameter watching to a model when enabled.

    :param Any run: Active W&B run or ``None``.
    :param object model: Model to watch.
    :param str log: W&B watch mode, typically ``gradients`` or ``all``.
    :param int log_freq: Histogram logging frequency in steps.
    """

    if run is None:
        return
    run.watch(model, log=log, log_freq=max(log_freq, 1), log_graph=False)


def wandb_log(run: Any, metrics: dict[str, Any], *, step: int) -> None:
    """Log one metrics payload to W&B when enabled.

    :param Any run: Active W&B run or ``None``.
    :param dict[str, Any] metrics: Metrics payload to log.
    :param int step: Global step for this payload.
    """

    if run is None:
        return
    run.log(metrics, step=step)


def wandb_summary_update(run: Any, metrics: dict[str, Any]) -> None:
    """Update the W&B run summary when enabled.

    :param Any run: Active W&B run or ``None``.
    :param dict[str, Any] metrics: Summary metrics to persist.
    """

    if run is None:
        return
    run.summary.update(metrics)


def finish_wandb(run: Any) -> None:
    """Finish an active W&B run.

    :param Any run: Active W&B run or ``None``.
    """

    if run is None:
        return
    run.finish()
