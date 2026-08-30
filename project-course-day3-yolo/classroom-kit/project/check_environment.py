import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))


def main() -> None:
    print("=== 課堂環境檢查 ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print(f"System: {platform.system()} / {platform.machine()}")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("[失敗] 尚未啟動 .venv")

    import cv2
    import ultralytics

    print(f"[成功] OpenCV {cv2.__version__}")
    print(f"[成功] Ultralytics {ultralytics.__version__}")
    for path in (ROOT / "images" / "horse.jpg", ROOT / "weights" / "yolo26n.pt"):
        if not path.is_file():
            raise SystemExit(f"[失敗] 找不到 {path}")
        print(f"[成功] 找到 {path.relative_to(ROOT)}")
    print("=== 可以開始上課 ===")


if __name__ == "__main__":
    main()
