from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

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


def _load_yolo_model(model_path: Path, device: str | None):
    from ultralytics import YOLO  # Local import to keep optional dependency lazy

    LOGGER.info("Loading YOLO model from %s", model_path)
    model = YOLO(str(model_path))
    if device:
        model.to(device)
    return model


def _yolo_detections_for_image(
    model,
    image_path: Path,
    allowed_classes: Sequence[str] | None,
    confidence_threshold: float,
) -> List[Detection]:
    detections: List[Detection] = []
    results = model.predict(source=str(image_path), conf=confidence_threshold, verbose=False)

    if not results:
        return detections

    result = results[0]
    class_filter = {cls.lower() for cls in allowed_classes} if allowed_classes else None
    names = getattr(result, "names", {}) or getattr(model, "names", {})

    width = int(result.orig_shape[1])
    height = int(result.orig_shape[0])

    for box in getattr(result, "boxes", []) or []:
        cls_idx = int(box.cls)
        label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)

        if class_filter and label.lower() not in class_filter:
            continue

        confidence = float(box.conf)
        x_min, y_min, x_max, y_max = [float(v) for v in box.xyxy[0].tolist()]

        detections.append(
            Detection(
                image_path=image_path,
                site=infer_site_from_name(image_path),
                label=label,
                confidence=confidence,
                x_min=max(0, int(round(x_min))),
                y_min=max(0, int(round(y_min))),
                x_max=min(width, int(round(x_max))),
                y_max=min(height, int(round(y_max))),
            )
        )

    return detections


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

    model = None
    confidence_threshold = float(detection_cfg.get("confidence_threshold", min_confidence))
    model_path = Path(detection_cfg.get("model_path", "models/waldo.pt"))
    device = detection_cfg.get("device")

    if not demo_mode:
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model weights not found at {model_path}")
        model = _load_yolo_model(model_path, device)

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
            if model is None:
                raise RuntimeError("YOLO model failed to load")
            detections.extend(
                _yolo_detections_for_image(
                    model,
                    image_path,
                    classes,
                    confidence_threshold,
                )
            )

    _save_thumbnails(detections, thumbnails_dir)
    _write_csv(detections, detections_path)

    LOGGER.info("Saved %s detections to %s", len(detections), detections_path)
    return detections_path
