"""教師完成版：OpenCV 讀圖與畫框，YOLO 推論，Counter 分類統計。"""

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
OUTPUT_PATH = ROOT / "outputs" / "final_result.jpg"
CONFIDENCE_THRESHOLD = 0.25


def main() -> None:
    image = cv2.imdecode(np.fromfile(IMAGE_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"OpenCV 無法讀取圖片：{IMAGE_PATH}")

    model = YOLO(str(MODEL_PATH))
    result = model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    counts: Counter[str] = Counter()

    print(f"圖片：{IMAGE_PATH.name}")
    for index, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        counts[class_name] += 1

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
        print(
            f"[{index}] {class_name}, confidence={confidence:.3f}, "
            f"box=({x1}, {y1}, {x2}, {y2})"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise OSError(f"無法編碼結果：{OUTPUT_PATH}")
    encoded.tofile(OUTPUT_PATH)

    print("\n分類統計：")
    for class_name, count in sorted(counts.items()):
        print(f"{class_name}: {count}")
    print(f"總物件數：{sum(counts.values())}")
    print(f"結果已儲存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
