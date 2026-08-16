import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / ".ultralytics"))

from ultralytics import YOLO  # noqa: E402

IMAGE_DIR = PROJECT_DIR / "images"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_PATH = PROJECT_DIR / "weights" / "yolo26n.pt"
CONFIDENCE_THRESHOLD = 0.25


def main() -> None:
    """Detect birds in every image and save annotated outputs."""
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weight not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(IMAGE_DIR), conf=CONFIDENCE_THRESHOLD, save=True,
        project=str(OUTPUT_DIR), name="bird_detection", exist_ok=True,
        verbose=False,
    )

    total_bird_count = 0
    for result in results:
        image_name = Path(result.path).name
        bird_count = 0
        print(f"\nImage: {image_name}")

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            if class_name != "bird":
                continue

            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bird_count += 1
            total_bird_count += 1
            print(
                f"  [{bird_count}] class={class_name}, confidence={confidence:.3f}, "
                f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )

        print(f"  Bird count: {bird_count}")

    print(f"\nTotal bird count: {total_bird_count}")
    print(f"Annotated outputs: {OUTPUT_DIR / 'bird_detection'}")


if __name__ == "__main__":
    main()
