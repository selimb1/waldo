from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from .utils import ensure_directories, infer_site_from_name

LOGGER = logging.getLogger(__name__)


@dataclass
class Detection:
    image_path: Path
    site: str
    label: str
    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    thumbnail_path: Path | None = None

    def to_row(self) -> List[str]:
        return [
            str(self.image_path),
            self.site,
            self.label,
            f"{self.confidence:.3f}",
            str(self.x_min),
            str(self.y_min),
            str(self.x_max),
            str(self.y_max),
            str(self.thumbnail_path) if self.thumbnail_path else "",
        ]


HEADER = [
    "image_path",
    "site",
    "class",
    "confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "thumbnail_path",
]


def _generate_demo_detections(
    image_path: Path,
    classes: Iterable[str],
    max_detections: int,
    min_confidence: float,
    max_confidence: float,
    rng: random.Random,
) -> List[Detection]:
    with Image.open(image_path) as img:
        width, height = img.size

    detections: List[Detection] = []
    num_detections = rng.randint(0, max_detections)
    LOGGER.debug("Demo detections for %s: %s", image_path.name, num_detections)

    for _ in range(num_detections):
        label = rng.choice(tuple(classes))
        confidence = rng.uniform(min_confidence, max_confidence)
        box_width = int(width * rng.uniform(0.1, 0.4))
        box_height = int(height * rng.uniform(0.1, 0.4))
        x_min = rng.randint(0, max(1, width - box_width))
        y_min = rng.randint(0, max(1, height - box_height))
        x_max = x_min + box_width
        y_max = y_min + box_height

        detections.append(
            Detection(
                image_path=image_path,
                site=infer_site_from_name(image_path),
                label=label,
                confidence=confidence,
                x_min=x_min,
                y_min=y_min,
                x_max=min(width, x_max),
                y_max=min(height, y_max),
            )
        )
    return detections


def _save_thumbnails(detections: Iterable[Detection], thumbnail_dir: Path) -> None:
    ensure_directories(thumbnail_dir)
    for idx, detection in enumerate(detections):
        with Image.open(detection.image_path) as img:
            cropped = img.crop((detection.x_min, detection.y_min, detection.x_max, detection.y_max))
            thumbnail_name = f"{detection.image_path.stem}_{idx:02d}.jpg"
            thumbnail_path = thumbnail_dir / thumbnail_name
            cropped.save(thumbnail_path, format="JPEG")
            detection.thumbnail_path = thumbnail_path


def _write_csv(detections: Iterable[Detection], output_path: Path) -> None:
    ensure_directories(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for detection in detections:
            writer.writerow(detection.to_row())


def run_detection(config: dict) -> Path:
    detection_cfg = config.get("detection", {})
    paths_cfg = config.get("paths", {})

    input_dir = Path(paths_cfg["inputs"])
    detections_path = Path(paths_cfg["detections"])
    thumbnails_dir = Path(paths_cfg["thumbnails"])

    ensure_directories(input_dir, detections_path.parent, thumbnails_dir)

    demo_mode = detection_cfg.get("demo_mode", True)
    classes = detection_cfg.get("classes", [])
    if not classes:
        raise ValueError("No classes configured for detection")

    rng = random.Random(detection_cfg.get("seed", 42))
    max_detections = int(detection_cfg.get("max_detections_per_image", 5))
    min_confidence = float(detection_cfg.get("min_confidence", 0.2))
    max_confidence = float(detection_cfg.get("max_confidence", 0.9))

    detections: List[Detection] = []

    image_files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    if not image_files:
        LOGGER.warning("No images found in %s", input_dir)

    for image_path in image_files:
        LOGGER.info("Processing image %s", image_path.name)
        if demo_mode:
            detections.extend(
                _generate_demo_detections(
                    image_path,
                    classes,
                    max_detections,
                    min_confidence,
                    max_confidence,
                    rng,
                )
            )
        else:
            raise NotImplementedError("Real YOLO detection not implemented in this MVP")

    _save_thumbnails(detections, thumbnails_dir)
    _write_csv(detections, detections_path)

    LOGGER.info("Saved %s detections to %s", len(detections), detections_path)
    return detections_path
