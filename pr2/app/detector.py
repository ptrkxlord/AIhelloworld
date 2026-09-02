import time

from ultralytics import YOLO

_model: YOLO | None = None


def load_model(weights: str = "yolov8n.pt") -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(weights)
    return _model


def detect(image_path: str, conf_threshold: float = 0.25) -> dict:
    model = load_model()

    start = time.perf_counter()
    results = model.predict(source=image_path, conf=conf_threshold, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = results[0]
    objects = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        objects.append(
            {
                "class": result.names[cls_id],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )

    return {
        "objects": objects,
        "count": len(objects),
        "inference_time_ms": round(elapsed_ms, 2),
    }
