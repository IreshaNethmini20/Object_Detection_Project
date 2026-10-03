"""Reusable CPU inference with OpenCV's Darknet importer."""
from collections import Counter
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TypedDict
import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'

class BoundingBox(TypedDict):
    x: int
    y: int
    width: int
    height: int

class Detection(TypedDict):
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox

class DetectionResult(TypedDict):
    total_detections: int
    counts: dict[str, int]
    detections: list[Detection]
    inference_time_ms: float

def load_image(path: str | Path) -> np.ndarray:
    """Read BGR pixels, supporting Unicode paths on Windows."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Image not found: {path}')
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image

class YOLOv4Detector:
    """Load one network; serialize mutable network operations across requests."""
    def __init__(self, model_dir: str | Path = MODEL_DIR, confidence: float = 0.5,
                 nms_threshold: float = 0.4, input_size: int = 416) -> None:
        if not 0 < confidence <= 1 or not 0 < nms_threshold <= 1:
            raise ValueError('Confidence and NMS thresholds must be in (0, 1].')
        if input_size <= 0 or input_size % 32:
            raise ValueError('Input size must be a positive multiple of 32.')
        self.confidence, self.nms_threshold, self.input_size = confidence, nms_threshold, input_size
        folder = Path(model_dir)
        cfg, weights, names = [folder / name for name in ('yolov4.cfg', 'yolov4.weights', 'coco.names')]
        for path in (cfg, weights, names):
            if not path.is_file():
                hint = ' Download pretrained weights; see models/README.md.' if path == weights else ''
                raise FileNotFoundError(f'Missing model file: {path}.{hint}')
        self.classes = names.read_text(encoding='utf-8').splitlines()
        if len(self.classes) != 80 or any(not name.strip() for name in self.classes):
            raise ValueError('coco.names must contain 80 nonempty COCO labels.')
        try:
            self.net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
        except cv2.error as exc:
            raise ValueError('Cannot load YOLOv4. Check that cfg and weights are valid and match.') from exc
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layers = self.net.getLayerNames()
        self.output_layers = [layers[int(i) - 1] for i in
                              np.asarray(self.net.getUnconnectedOutLayers()).reshape(-1)]
        self._lock = Lock()

    def detect(self, image: np.ndarray | str | Path) -> DetectionResult:
        """Detect in original pixel coordinates; time only setInput and forward."""
        if isinstance(image, (str, Path)):
            image = load_image(image)
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
                or image.shape[2] != 3 or image.size == 0):
            raise ValueError('Expected a nonempty uint8 BGR image with three channels.')
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.input_size, self.input_size),
                                    swapRB=True, crop=False)
        with self._lock:
            start = perf_counter()
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
            elapsed = (perf_counter() - start) * 1000
        detections = self._postprocess(outputs, image.shape[1], image.shape[0])
        return {'total_detections': len(detections),
                'counts': dict(sorted(Counter(d['class_name'] for d in detections).items())),
                'detections': detections, 'inference_time_ms': round(elapsed, 2)}

    def _postprocess(self, outputs: list[np.ndarray], width: int, height: int) -> list[Detection]:
        """Filter OpenCV Region scores, then suppress overlaps within each class."""
        boxes, scores, class_ids = [], [], []
        for output in outputs:
            for row in output:
                if not np.isfinite(row).all():
                    continue
                class_id = int(np.argmax(row[5:]))
                # OpenCV Region already multiplies objectness by class probability.
                # row[5:] contains joint scores; do not multiply by row[4] again.
                score = float(row[5 + class_id])
                if score <= self.confidence:
                    continue
                cx, cy, bw, bh = row[:4] * [width, height, width, height]
                x1 = max(0, min(width, int(round(cx - bw / 2))))
                y1 = max(0, min(height, int(round(cy - bh / 2))))
                x2 = max(0, min(width, int(round(cx + bw / 2))))
                y2 = max(0, min(height, int(round(cy + bh / 2))))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(score)
                class_ids.append(class_id)
        kept = []
        for class_id in sorted(set(class_ids)):
            group = [i for i, value in enumerate(class_ids) if value == class_id]
            indices = cv2.dnn.NMSBoxes([boxes[i] for i in group], [scores[i] for i in group],
                                       self.confidence, self.nms_threshold)
            kept.extend(group[int(i)] for i in np.asarray(indices).reshape(-1))
        return [{'class_id': class_ids[i], 'class_name': self.classes[class_ids[i]],
                 'confidence': scores[i],
                 'bounding_box': dict(zip(('x', 'y', 'width', 'height'), boxes[i]))}
                for i in sorted(kept, key=lambda i: scores[i], reverse=True)]

    @staticmethod
    def draw_boxes(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Annotate a copy, preserving original image dimensions."""
        annotated = image.copy()
        for detection in detections:
            box = detection['bounding_box']
            x, y, w, h = [box[key] for key in ('x', 'y', 'width', 'height')]
            cv2.rectangle(annotated, (x, y), (x + w - 1, y + h - 1), (0, 220, 0), 2)
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x = max(0, min(x, image.shape[1] - tw - 4))
            bottom = min(image.shape[0] - 1, max(th + baseline + 4, y))
            cv2.rectangle(annotated, (label_x, bottom - th - baseline - 4),
                          (label_x + tw + 4, bottom), (0, 220, 0), -1)
            cv2.putText(annotated, label, (label_x + 2, bottom - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return annotated

    def save_output(self, image: np.ndarray, detections: list[Detection], path: str | Path) -> Path:
        """Save an annotated image, supporting Unicode paths on Windows."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(path.suffix, self.draw_boxes(image, detections))
        if not success:
            raise OSError(f'Cannot encode output: {path}')
        path.write_bytes(encoded.tobytes())
        return path
