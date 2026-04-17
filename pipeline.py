"""Two-stage license plate pipeline: YOLO detector -> FastPlateOCR recognizer.

Both models run via onnxruntime (CPU) so this works on Raspberry Pi 4.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from fast_plate_ocr import LicensePlateRecognizer
from open_image_models import LicensePlateDetector


@dataclass
class PlateResult:
    text: str
    confidence: float  # mean char confidence from OCR
    det_confidence: float  # YOLO detection confidence
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


class PlatePipeline:
    def __init__(
        self,
        ocr_onnx_path: Path,
        plate_config_path: Path,
        detector_model: str = "yolo-v9-t-384-license-plate-end2end",
        min_det_conf: float = 0.35,
        padding: int = 4,
    ):
        self.detector = LicensePlateDetector(detection_model=detector_model)
        self.recognizer = LicensePlateRecognizer(
            onnx_model_path=str(ocr_onnx_path),
            plate_config_path=str(plate_config_path),
        )
        cfg = yaml.safe_load(Path(plate_config_path).read_text(encoding="utf-8"))
        self.pad_char = cfg["pad_char"]
        self.min_det_conf = min_det_conf
        self.padding = padding

    def _crop(self, img: np.ndarray, box) -> np.ndarray:
        h, w = img.shape[:2]
        x1 = max(0, box.x1 - self.padding)
        y1 = max(0, box.y1 - self.padding)
        x2 = min(w, box.x2 + self.padding)
        y2 = min(h, box.y2 + self.padding)
        return img[y1:y2, x1:x2], (x1, y1, x2, y2)

    def run(self, bgr_image: np.ndarray) -> list[PlateResult]:
        """Detect and recognize every plate in a BGR image. Returns list of results."""
        if bgr_image is None or bgr_image.size == 0:
            return []

        detections = self.detector.predict(bgr_image) or []
        results: list[PlateResult] = []

        for det in detections:
            if det.confidence < self.min_det_conf:
                continue
            crop, bbox = self._crop(bgr_image, det.bounding_box)
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 20:
                continue

            # LicensePlateRecognizer expects a numpy image or path; it handles preprocessing.
            preds = self.recognizer.run(crop, return_confidence=True)
            if not preds:
                continue
            p = preds[0]
            text = (p.plate or "").rstrip(self.pad_char)
            conf = float(np.mean(p.char_probs)) if p.char_probs is not None else 0.0

            results.append(
                PlateResult(
                    text=text,
                    confidence=conf,
                    det_confidence=float(det.confidence),
                    bbox=bbox,
                )
            )
        return results

    @staticmethod
    def draw(bgr_image: np.ndarray, results: list[PlateResult]) -> np.ndarray:
        """Return a copy of bgr_image with boxes + plate text drawn."""
        out = bgr_image.copy()
        for r in results:
            x1, y1, x2, y2 = r.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{r.text} ({r.confidence:.2f})" if r.text else f"plate ({r.det_confidence:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(out, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        return out
