"""學生起始版：依 TODO 完成多物件分類統計與 OpenCV 畫框。"""

import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO  # noqa: E402

MODEL_PATH = ROOT / "weights" / "yolo26n.pt"
IMAGE_PATH = ROOT / "images" / "horse.jpg"
OUTPUT_PATH = ROOT / "outputs" / "student_result.jpg"
CONFIDENCE_THRESHOLD = 0.25

image = cv2.imdecode(np.fromfile(IMAGE_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError(f"無法讀取圖片：{IMAGE_PATH}")

model = YOLO(str(MODEL_PATH))
result = model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
counts = Counter()

for box in result.boxes:
    # TODO 1：從 box.cls 取得 class_id，再由 result.names 取得 class_name。
    class_id = int(box.cls.item())
    class_name = result.names[class_id]

    # TODO 2：取得 confidence 和 xyxy 四個座標。
    confidence = float(box.conf.item())
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    # TODO 3：讓目前類別的數量加一。
    counts[class_name] += 1

    # TODO 4：用 OpenCV 畫框並寫上類別與 confidence。
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"{class_name} {confidence:.2f}",
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
success, encoded = cv2.imencode(".jpg", image)
if not success:
    raise OSError("OpenCV 無法編碼輸出圖片")
encoded.tofile(OUTPUT_PATH)

print(f"圖片：{IMAGE_PATH.name}")
for class_name, count in sorted(counts.items()):
    print(f"{class_name}: {count}")
print(f"總物件數：{sum(counts.values())}")
print(f"結果：{OUTPUT_PATH}")
