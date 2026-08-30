from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "images" / "horse.jpg"
OUTPUT_PATH = ROOT / "outputs" / "step1_opencv.jpg"

# 先讀成位元組再交給 OpenCV 解碼，可同時支援中英文路徑。
image = cv2.imdecode(np.fromfile(IMAGE_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError(f"OpenCV 無法讀取圖片：{IMAGE_PATH}")

height, width, channels = image.shape
print(f"圖片：{IMAGE_PATH.name}")
print(f"寬={width}, 高={height}, 色彩通道={channels}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
success, encoded = cv2.imencode(".jpg", image)
if not success:
    raise OSError("OpenCV 無法編碼輸出圖片")
encoded.tofile(OUTPUT_PATH)
print(f"已儲存：{OUTPUT_PATH}")
