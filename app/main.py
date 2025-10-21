from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .detect import run_detection
from .reconcile import run_reconciliation
from .report import render_report
from .utils import ensure_directories, load_config, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WALDO Audit MVP pipeline")
    parser.add_argument(
        "--config",
        default=Path("app/config.yaml"),
        type=Path,
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    paths_cfg = config.get("paths", {})
    ensure_directories(paths_cfg.get("outputs", "data/outputs"))

    LOGGER.info("Starting WALDO Audit pipeline")
    detections_path = run_detection(config)
    reconciliation_path = run_reconciliation(config, detections_path)
    render_report(config, detections_path, reconciliation_path)
    LOGGER.info("Pipeline finished successfully")


if __name__ == "__main__":  # pragma: no cover
    main()
