from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def ensure_directories(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_inventory(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"site", "class", "declared_quantity"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Inventory file is missing columns: {missing}")
    return df


def load_sites(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"site", "latitude", "longitude"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Sites file is missing columns: {missing}")
    return df


def infer_site_from_name(image_path: Path) -> str:
    """Infer site name from filename prefix before first underscore."""
    stem = image_path.stem
    if "_" in stem:
        return stem.split("_")[0]
    return stem
