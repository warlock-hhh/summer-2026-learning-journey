"""預覽同一張 Food-11 圖片經過隨機 Data Augmentation 後的結果。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT / "food-11" / "training" / "labeled"
OUTPUT_PATH = ROOT / "augmentation_preview.jpg"

# 這組 transform 只用於預覽，尚未套用到正式 baseline 訓練。
# 不加入 ToTensor，因為我們要保留 PIL Image 並輸出成 JPG。
AUGMENT = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            128,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10, fill=(128, 128, 128)),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.1,
            hue=0.02,
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="預覽 Food-11 data augmentation")
    parser.add_argument(
        "--image",
        type=Path,
        help="指定一張 JPG；未指定時會從 training/labeled 隨機選取",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def choose_image(specified_path: Path | None) -> Path:
    if specified_path is not None:
        image_path = specified_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"找不到圖片：{image_path}")
        return image_path

    candidates = list(TRAIN_ROOT.glob("*/*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"找不到 training JPG：{TRAIN_ROOT}")
    return random.choice(candidates)


def add_tile(
    canvas: Image.Image,
    image: Image.Image,
    label: str,
    position: int,
) -> None:
    columns = 4
    tile_width = 148
    tile_height = 166
    margin = 10

    row, column = divmod(position, columns)
    x = margin + column * tile_width
    y = margin + row * tile_height

    resized = image.resize((128, 128))
    canvas.paste(resized, (x, y + 20))
    ImageDraw.Draw(canvas).text((x, y), label, fill="black")


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    image_path = choose_image(args.image)
    with Image.open(image_path) as source:
        original = source.convert("RGB")

    # 4×4：左上角為原圖，其餘 15 格每次重新隨機增強。
    canvas = Image.new("RGB", (4 * 148 + 20, 4 * 166 + 20), "white")
    add_tile(canvas, original, "Original", 0)
    for index in range(1, 16):
        add_tile(canvas, AUGMENT(original), f"Aug {index:02d}", index)

    canvas.save(OUTPUT_PATH, quality=95)
    print(f"Source: {image_path}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
