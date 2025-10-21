from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

try:
    import folium
except ImportError as exc:  # pragma: no cover
    folium = None
    FOLIUM_IMPORT_ERROR = exc
else:
    FOLIUM_IMPORT_ERROR = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except ImportError as exc:  # pragma: no cover
    SimpleDocTemplate = None
    PDF_IMPORT_ERROR = exc
else:
    PDF_IMPORT_ERROR = None

from .utils import ensure_directories, load_sites


def _determine_value_status(difference: float, tolerance: float = 1e-6) -> str:
    if abs(difference) <= tolerance:
        return "OK"
    if difference > 0:
        return "SOBRANTE"
    return "FALTANTE"

LOGGER = logging.getLogger(__name__)


def _format_currency(value: float | int | None) -> str:
    if value is None:
        return "$0.00"
    return f"${float(value):,.2f}"


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
            popup_html += (
                "<tr>"
                "<th>Clase</th><th>Declarado</th><th>Detectado</th><th>Diferencia</th><th>Estado</th>"
                "<th>Valor decl.</th><th>Valor det.</th><th>Dif. valor</th><th>Estado valor</th>"
                "<th>Val. dep. decl.</th><th>Val. dep. det.</th><th>Dif. dep.</th><th>Estado dep.</th>"
                "</tr>"
            )
            for _, rec in records.iterrows():
                declared_gross = rec.get("declared_gross_value", 0.0)
                detected_gross = rec.get("detected_gross_value", 0.0)
                gross_diff = rec.get("gross_value_difference", 0.0)
                declared_residual = rec.get("declared_residual_value", 0.0)
                detected_residual = rec.get("detected_residual_value", 0.0)
                residual_diff = rec.get("residual_value_difference", 0.0)
                popup_html += (
                    "<tr>"
                    f"<td>{rec['class']}</td>"
                    f"<td>{rec['declared_quantity']}</td>"
                    f"<td>{rec['detected_quantity']}</td>"
                    f"<td>{rec['difference']}</td>"
                    f"<td>{rec['status']}</td>"
                    f"<td>{_format_currency(declared_gross)}</td>"
                    f"<td>{_format_currency(detected_gross)}</td>"
                    f"<td>{_format_currency(gross_diff)}</td>"
                    f"<td>{rec.get('gross_value_status', 'OK')}</td>"
                    f"<td>{_format_currency(declared_residual)}</td>"
                    f"<td>{_format_currency(detected_residual)}</td>"
                    f"<td>{_format_currency(residual_diff)}</td>"
                    f"<td>{rec.get('residual_value_status', 'OK')}</td>"
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
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    report_cfg = config.get("report", {})

    if not detections_df.empty and "confidence" in detections_df.columns:
        detections_df["confidence"] = detections_df["confidence"].astype(float)
    if not reconciliation_df.empty:
        quantity_columns = ["declared_quantity", "detected_quantity", "difference"]
        for column in quantity_columns:
            if column in reconciliation_df.columns:
                reconciliation_df[column] = reconciliation_df[column].astype(int)

        value_columns = [
            "declared_gross_value",
            "detected_gross_value",
            "gross_value_difference",
            "declared_residual_value",
            "detected_residual_value",
            "residual_value_difference",
        ]
        for column in value_columns:
            if column in reconciliation_df.columns:
                reconciliation_df[column] = reconciliation_df[column].astype(float)

    summary_df = (
        reconciliation_df.groupby(["site", "status"]).size().reset_index(name="count")
        if not reconciliation_df.empty
        else pd.DataFrame(columns=["site", "status", "count"])
    )

    value_summary_df = pd.DataFrame()
    if not reconciliation_df.empty:
        grouped = reconciliation_df.groupby("site")
        value_summary_df = grouped[
            [
                "declared_gross_value",
                "detected_gross_value",
                "gross_value_difference",
                "declared_residual_value",
                "detected_residual_value",
                "residual_value_difference",
            ]
        ].sum()
        value_summary_df = value_summary_df.reset_index()
        value_summary_df["gross_value_status"] = value_summary_df[
            "gross_value_difference"
        ].apply(_determine_value_status)
        value_summary_df["residual_value_status"] = value_summary_df[
            "residual_value_difference"
        ].apply(_determine_value_status)
        value_summary_df = value_summary_df[
            [
                "site",
                "declared_gross_value",
                "detected_gross_value",
                "gross_value_difference",
                "gross_value_status",
                "declared_residual_value",
                "detected_residual_value",
                "residual_value_difference",
                "residual_value_status",
            ]
        ]

    total_detected = (
        int(reconciliation_df["detected_quantity"].sum()) if not reconciliation_df.empty else 0
    )
    total_declared = (
        int(reconciliation_df["declared_quantity"].sum()) if not reconciliation_df.empty else 0
    )

    total_declared_value = (
        float(reconciliation_df["declared_gross_value"].sum())
        if "declared_gross_value" in reconciliation_df
        else 0.0
    )
    total_detected_value = (
        float(reconciliation_df["detected_gross_value"].sum())
        if "detected_gross_value" in reconciliation_df
        else 0.0
    )
    total_declared_residual_value = (
        float(reconciliation_df["declared_residual_value"].sum())
        if "declared_residual_value" in reconciliation_df
        else 0.0
    )
    total_detected_residual_value = (
        float(reconciliation_df["detected_residual_value"].sum())
        if "detected_residual_value" in reconciliation_df
        else 0.0
    )

    total_gross_difference = (
        float(reconciliation_df["gross_value_difference"].sum())
        if "gross_value_difference" in reconciliation_df
        else 0.0
    )
    total_residual_difference = (
        float(reconciliation_df["residual_value_difference"].sum())
        if "residual_value_difference" in reconciliation_df
        else 0.0
    )

    map_html = _build_map_html(sites_df, reconciliation_df)

    context = {
        "title": report_cfg.get("title", "Informe de Auditoría"),
        "company_name": report_cfg.get("company_name", ""),
        "detections": detections_df.to_dict(orient="records"),
        "reconciliation": reconciliation_df.to_dict(orient="records"),
        "summary": summary_df.to_dict(orient="records"),
        "value_summary": value_summary_df.to_dict(orient="records"),
        "total_detected": total_detected,
        "total_declared": total_declared,
        "total_declared_value": total_declared_value,
        "total_detected_value": total_detected_value,
        "total_gross_value_difference": total_gross_difference,
        "total_declared_residual_value": total_declared_residual_value,
        "total_detected_residual_value": total_detected_residual_value,
        "total_residual_value_difference": total_residual_difference,
        "map_html": map_html,
    }
    return context, summary_df, value_summary_df


def _export_pdf(
    context: Dict[str, Any],
    reconciliation_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    value_summary_df: pd.DataFrame,
    pdf_path: Path,
) -> None:
    ensure_directories(pdf_path.parent)

    if SimpleDocTemplate is None:
        LOGGER.warning("ReportLab is not available: %s", PDF_IMPORT_ERROR)
        return

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    title = context.get("title", "Informe de Auditoría")
    elements.append(Paragraph(title, styles["Title"]))

    company = context.get("company_name")
    if company:
        elements.append(Paragraph(company, styles["Heading2"]))

    elements.append(Spacer(1, 12))
    totals_text = (
        f"Total declarado: {context.get('total_declared', 0)} | "
        f"Total detectado: {context.get('total_detected', 0)}"
    )
    elements.append(Paragraph(totals_text, styles["Normal"]))

    declared_value = context.get("total_declared_value", 0.0)
    detected_value = context.get("total_detected_value", 0.0)
    if declared_value or detected_value:
        value_totals = (
            f"Valor declarado: ${declared_value:,.2f} | "
            f"Valor detectado: ${detected_value:,.2f}"
        )
        elements.append(Paragraph(value_totals, styles["Normal"]))

    declared_residual = context.get("total_declared_residual_value", 0.0)
    detected_residual = context.get("total_detected_residual_value", 0.0)
    if declared_residual or detected_residual:
        residual_totals = (
            f"Valor depreciado declarado: ${declared_residual:,.2f} | "
            f"Valor depreciado detectado: ${detected_residual:,.2f}"
        )
        elements.append(Paragraph(residual_totals, styles["Normal"]))
    elements.append(Spacer(1, 18))

    if not summary_df.empty:
        summary_data = [["Sitio", "Estado", "Cantidad"]] + summary_df.astype(str).values.tolist()
        summary_table = Table(summary_data, hAlign="LEFT")
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ]
            )
        )
        elements.append(Paragraph("Resumen por sitio", styles["Heading3"]))
        elements.append(summary_table)
        elements.append(Spacer(1, 18))

    if not reconciliation_df.empty:
        recon_data = [list(reconciliation_df.columns)] + reconciliation_df.astype(str).values.tolist()
        recon_table = Table(recon_data, hAlign="LEFT")
        recon_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ]
            )
        )
        elements.append(Paragraph("Conciliación", styles["Heading3"]))
        elements.append(recon_table)
        elements.append(Spacer(1, 18))

    if not detections_df.empty:
        det_data = [list(detections_df.columns)] + detections_df.astype(str).values.tolist()
        det_table = Table(det_data, hAlign="LEFT")
        det_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ]
            )
        )
        elements.append(Paragraph("Detecciones", styles["Heading3"]))
        elements.append(det_table)

    if not value_summary_df.empty:
        value_summary_data = [
            [
                "Sitio",
                "Valor declarado",
                "Valor detectado",
                "Diferencia",
                "Estado valor",
                "Val. dep. declarado",
                "Val. dep. detectado",
                "Dif. dep.",
                "Estado dep.",
            ]
        ] + value_summary_df.astype(str).values.tolist()
        value_table = Table(value_summary_data, hAlign="LEFT")
        value_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ]
            )
        )
        elements.append(Paragraph("Resumen de valores por obra", styles["Heading3"]))
        elements.append(value_table)
        elements.append(Spacer(1, 18))

    try:
        doc.build(elements)
    except Exception as exc:  # pragma: no cover - reportlab internal errors
        LOGGER.error("Failed to export PDF report to %s: %s", pdf_path, exc)
        return

    LOGGER.info("PDF report saved to %s", pdf_path)


def _export_excel(
    reconciliation_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    value_summary_df: pd.DataFrame,
    excel_path: Path,
) -> None:
    ensure_directories(excel_path.parent)

    try:
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            recon_export = (
                reconciliation_df
                if not reconciliation_df.empty
                else pd.DataFrame([{"mensaje": "Sin datos de conciliación"}])
            )
            recon_export.to_excel(writer, sheet_name="Conciliacion", index=False)

            detections_export = (
                detections_df
                if not detections_df.empty
                else pd.DataFrame([{"mensaje": "Sin detecciones disponibles"}])
            )
            detections_export.to_excel(writer, sheet_name="Detecciones", index=False)

            summary_export = (
                summary_df
                if not summary_df.empty
                else pd.DataFrame([{"mensaje": "Sin resumen disponible"}])
            )
            summary_export.to_excel(writer, sheet_name="Resumen", index=False)

            value_summary_export = (
                value_summary_df
                if not value_summary_df.empty
                else pd.DataFrame([{"mensaje": "Sin resumen de valores"}])
            )
            value_summary_export.to_excel(writer, sheet_name="ResumenValores", index=False)
    except ImportError as exc:
        LOGGER.warning("Unable to export Excel report due to missing dependency: %s", exc)
        return

    LOGGER.info("Excel report saved to %s", excel_path)


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

    context, summary_df, value_summary_df = _prepare_context(
        config, detections_df, reconciliation_df, sites_df
    )

    html_content = template.render(**context)
    report_path.write_text(html_content, encoding="utf-8")
    LOGGER.info("Report saved to %s", report_path)

    pdf_cfg = paths_cfg.get("report_pdf")
    excel_cfg = paths_cfg.get("report_excel")

    if pdf_cfg:
        _export_pdf(
            context,
            reconciliation_df,
            detections_df,
            summary_df,
            value_summary_df,
            Path(pdf_cfg),
        )

    if excel_cfg:
        _export_excel(
            reconciliation_df,
            detections_df,
            summary_df,
            value_summary_df,
            Path(excel_cfg),
        )

    return report_path
