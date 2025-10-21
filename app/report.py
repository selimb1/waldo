from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

try:
    import folium
except ImportError as exc:  # pragma: no cover
    folium = None
    FOLIUM_IMPORT_ERROR = exc
else:
    FOLIUM_IMPORT_ERROR = None

from .utils import ensure_directories, load_sites

LOGGER = logging.getLogger(__name__)


def _load_template(template_path: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(template_path.parent),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env


def _build_map_html(sites_df: pd.DataFrame, reconciliation_df: pd.DataFrame) -> str | None:
    if folium is None:
        LOGGER.warning("Folium is not available: %s", FOLIUM_IMPORT_ERROR)
        return None

    if sites_df.empty:
        LOGGER.warning("Sites dataframe is empty; skipping map generation")
        return None

    avg_lat = sites_df["latitude"].mean()
    avg_lon = sites_df["longitude"].mean()
    fmap = folium.Map(location=[avg_lat, avg_lon], zoom_start=5)

    for _, site_row in sites_df.iterrows():
        site = site_row["site"]
        records = reconciliation_df[reconciliation_df["site"] == site]
        if records.empty:
            popup_html = "<p>Sin datos de conciliación.</p>"
        else:
            popup_html = "<table class='table table-sm'>"
            popup_html += "<tr><th>Clase</th><th>Declarado</th><th>Detectado</th><th>Estado</th></tr>"
            for _, rec in records.iterrows():
                popup_html += (
                    "<tr>"
                    f"<td>{rec['class']}</td>"
                    f"<td>{rec['declared_quantity']}</td>"
                    f"<td>{rec['detected_quantity']}</td>"
                    f"<td>{rec['status']}</td>"
                    "</tr>"
                )
            popup_html += "</table>"
        folium.Marker(
            location=[site_row["latitude"], site_row["longitude"]],
            popup=folium.Popup(popup_html, max_width=400),
            tooltip=site_row.get("address", site),
        ).add_to(fmap)

    return fmap.get_root().render()


def _prepare_context(
    config: Dict[str, Any],
    detections_df: pd.DataFrame,
    reconciliation_df: pd.DataFrame,
    sites_df: pd.DataFrame,
) -> Dict[str, Any]:
    report_cfg = config.get("report", {})

    if not detections_df.empty and "confidence" in detections_df.columns:
        detections_df["confidence"] = detections_df["confidence"].astype(float)
    if not reconciliation_df.empty:
        for column in ["declared_quantity", "detected_quantity", "difference"]:
            if column in reconciliation_df.columns:
                reconciliation_df[column] = reconciliation_df[column].astype(int)

    summary = (
        reconciliation_df.groupby(["site", "status"]).size().reset_index(name="count")
        if not reconciliation_df.empty
        else pd.DataFrame(columns=["site", "status", "count"])
    )

    total_detected = int(reconciliation_df["detected_quantity"].sum()) if not reconciliation_df.empty else 0
    total_declared = int(reconciliation_df["declared_quantity"].sum()) if not reconciliation_df.empty else 0

    map_html = _build_map_html(sites_df, reconciliation_df)

    context = {
        "title": report_cfg.get("title", "Informe de Auditoría"),
        "company_name": report_cfg.get("company_name", ""),
        "detections": detections_df.to_dict(orient="records"),
        "reconciliation": reconciliation_df.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "total_detected": total_detected,
        "total_declared": total_declared,
        "map_html": map_html,
    }
    return context


def render_report(config: Dict[str, Any], detections_path: Path, reconciliation_path: Path) -> Path:
    paths_cfg = config.get("paths", {})
    template_path = Path(paths_cfg["template"])
    report_path = Path(paths_cfg["report"])
    ensure_directories(report_path.parent)

    template_env = _load_template(template_path)
    template = template_env.get_template(template_path.name)

    detections_df = pd.read_csv(detections_path) if detections_path.exists() else pd.DataFrame()
    reconciliation_df = (
        pd.read_csv(reconciliation_path) if reconciliation_path.exists() else pd.DataFrame()
    )
    sites_df = load_sites(paths_cfg["sites"])

    context = _prepare_context(config, detections_df, reconciliation_df, sites_df)

    html_content = template.render(**context)
    report_path.write_text(html_content, encoding="utf-8")
    LOGGER.info("Report saved to %s", report_path)
    return report_path
