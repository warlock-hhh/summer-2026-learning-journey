import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO  # noqa: E402

MODEL_PATH = ROOT / "weights" / "yolo26n.pt"
IMAGE_PATH = ROOT / "images" / "horse.jpg"

model = YOLO(str(MODEL_PATH))
results = model.predict(source=str(IMAGE_PATH), conf=0.25, verbose=False)
result = results[0]

for index, box in enumerate(result.boxes, start=1):
    class_id = int(box.cls.item())
    class_name = result.names[class_id]
    confidence = float(box.conf.item())
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(
        f"[{index}] class={class_name}, confidence={confidence:.3f}, "
        f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
    )
