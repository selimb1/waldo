from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .utils import ensure_directories, load_inventory

LOGGER = logging.getLogger(__name__)


STATUS_OK = "OK"
STATUS_SURPLUS = "SOBRANTE"
STATUS_MISSING = "FALTANTE"

VALUE_TOLERANCE = 1e-6


def _determine_status(difference: int) -> str:
    if difference == 0:
        return STATUS_OK
    if difference > 0:
        return STATUS_SURPLUS
    return STATUS_MISSING


def _determine_value_status(difference: float) -> str:
    if abs(difference) <= VALUE_TOLERANCE:
        return STATUS_OK
    if difference > 0:
        return STATUS_SURPLUS
    return STATUS_MISSING


def run_reconciliation(config: dict, detections_path: Path) -> Path:
    paths_cfg = config.get("paths", {})
    reconciliation_path = Path(paths_cfg["reconciliation"])
    inventory_path = Path(paths_cfg["inventory"])

    ensure_directories(reconciliation_path.parent)

    if detections_path.exists():
        detections_df = pd.read_csv(detections_path)
    else:
        LOGGER.warning("Detections file %s not found. Assuming no detections.", detections_path)
        detections_df = pd.DataFrame(columns=["site", "class", "confidence"])

    if detections_df.empty:
        detected_counts = pd.DataFrame(columns=["site", "class", "detected_quantity"])
    else:
        detected_counts = (
            detections_df.groupby(["site", "class"]).size().reset_index(name="detected_quantity")
        )

    inventory_df = load_inventory(inventory_path)

    reconciliation_df = pd.merge(
        inventory_df,
        detected_counts,
        on=["site", "class"],
        how="outer",
    )

    reconciliation_df["declared_quantity"] = (
        pd.to_numeric(reconciliation_df["declared_quantity"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    reconciliation_df["detected_quantity"] = (
        pd.to_numeric(reconciliation_df["detected_quantity"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    for column in ["declared_unit_value", "declared_unit_residual_value"]:
        if column not in reconciliation_df.columns:
            reconciliation_df[column] = 0.0
        reconciliation_df[column] = reconciliation_df[column].fillna(0).astype(float)

    reconciliation_df["difference"] = (
        reconciliation_df["detected_quantity"] - reconciliation_df["declared_quantity"]
    )
    reconciliation_df["status"] = reconciliation_df["difference"].apply(_determine_status)

    reconciliation_df["declared_gross_value"] = (
        reconciliation_df["declared_quantity"] * reconciliation_df["declared_unit_value"]
    )
    reconciliation_df["detected_gross_value"] = (
        reconciliation_df["detected_quantity"] * reconciliation_df["declared_unit_value"]
    )
    reconciliation_df["gross_value_difference"] = (
        reconciliation_df["detected_gross_value"] - reconciliation_df["declared_gross_value"]
    )
    reconciliation_df["gross_value_status"] = reconciliation_df["gross_value_difference"].apply(
        _determine_value_status
    )

    reconciliation_df["declared_residual_value"] = (
        reconciliation_df["declared_quantity"]
        * reconciliation_df["declared_unit_residual_value"]
    )
    reconciliation_df["detected_residual_value"] = (
        reconciliation_df["detected_quantity"]
        * reconciliation_df["declared_unit_residual_value"]
    )
    reconciliation_df["residual_value_difference"] = (
        reconciliation_df["detected_residual_value"]
        - reconciliation_df["declared_residual_value"]
    )
    reconciliation_df["residual_value_status"] = reconciliation_df[
        "residual_value_difference"
    ].apply(_determine_value_status)

    reconciliation_df = reconciliation_df.sort_values(["site", "class"]).reset_index(drop=True)

    reconciliation_df.to_csv(reconciliation_path, index=False)
    LOGGER.info("Saved reconciliation results to %s", reconciliation_path)
    return reconciliation_path
