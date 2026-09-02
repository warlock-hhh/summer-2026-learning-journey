from __future__ import annotations

"""
ML2021 HW3 學習版：先完整理解 supervised CNN，再進入進階技巧。

這份程式刻意只保留一條主線：
Data -> Model -> Logits -> Loss -> Gradient -> Optimizer -> Validation

執行範例：
1. 只觀察每一層的 Tensor shape：
   python hw03_learning.py --show-shapes
2. 跑 1 epoch 確認完整流程：
   python hw03_learning.py --epochs 1
3. 理解 baseline 後再開啟保守 augmentation：
   python hw03_learning.py --epochs 20 --augment
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "food-11"


def load_rgb_image(path: str) -> Image.Image:
    """讀取圖片，並統一轉成三個 channel 的 RGB。"""

    with Image.open(path) as image:
        return image.convert("RGB")


class Classifier(nn.Module):
    """官方 Easy baseline：輸入圖片，輸出 11 個尚未 Softmax 的 logits。"""

    def __init__(self) -> None:
        super().__init__()

        # CNN 的工作是從像素中抽取空間特徵。
        self.cnn_layers = nn.Sequential(
            # [B, 3, 128, 128] -> [B, 64, 128, 128]
            # 3：RGB channel；64：希望卷積學到的 64 種特徵。
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            # 穩定每個 channel 的數值分布，讓訓練較穩定。
            nn.BatchNorm2d(64),
            # 非線性函數：負數變 0，正數保留。
            nn.ReLU(),
            # 長寬各縮小一半：[B, 64, 128, 128] -> [B, 64, 64, 64]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二組卷積：channel 增加、圖片長寬繼續縮小。
            # -> [B, 128, 64, 64] -> pooling 後 [B, 128, 32, 32]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三組卷積：-> [B, 256, 32, 32]
            # 4x4 pooling 後得到 [B, 256, 8, 8]。
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )

        # Fully connected layers 根據 CNN 抽出的特徵做最後分類。
        self.fc_layers = nn.Sequential(
            # 256 * 8 * 8 = 16,384 個特徵，先壓縮成 256 維。
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # Food-11 有 11 類，所以最後輸出 11 個 logits。
            nn.Linear(256, 11),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """定義一個 batch 的圖片如何依序通過模型。"""

        features = self.cnn_layers(images)
        # 保留第 0 維 batch，把每張圖片的其餘維度攤平成一維。
        flattened_features = features.flatten(1)
        logits = self.fc_layers(flattened_features)
        # 這裡不做 Softmax；CrossEntropyLoss 內部會正確處理 logits。
        return logits


def build_loaders(
    batch_size: int,
    num_workers: int,
    augment: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """建立 training、validation、testing 三份 DataLoader。"""

    # Training 可以選擇加入保守的資料增強。
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (128, 128),
                    scale=(0.8, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )

    # Validation 與 testing 不可使用隨機增強，才能穩定比較。
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    train_set = DatasetFolder(
        DATA_ROOT / "training" / "labeled",
        loader=load_rgb_image,
        extensions=("jpg",),
        transform=train_transform,
    )
    valid_set = DatasetFolder(
        DATA_ROOT / "validation",
        loader=load_rgb_image,
        extensions=("jpg",),
        transform=evaluation_transform,
    )
    test_set = DatasetFolder(
        DATA_ROOT / "testing",
        loader=load_rgb_image,
        extensions=("jpg",),
        transform=evaluation_transform,
    )

    common_loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        **common_loader_options,
    )
    valid_loader = DataLoader(
        valid_set,
        shuffle=False,
        **common_loader_options,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        **common_loader_options,
    )
    return train_loader, valid_loader, test_loader


def show_tensor_shapes(model: Classifier, device: torch.device) -> None:
    """用一張假的圖片走過模型，觀察每一層如何改變 Tensor shape。"""

    tensor = torch.zeros(1, 3, 128, 128, device=device)
    print(f"Input{'':20s} -> {tuple(tensor.shape)}")

    for layer in model.cnn_layers:
        tensor = layer(tensor)
        print(f"{layer.__class__.__name__:25s} -> {tuple(tensor.shape)}")

    tensor = tensor.flatten(1)
    print(f"Flatten{'':18s} -> {tuple(tensor.shape)}")
    for layer in model.fc_layers:
        tensor = layer(tensor)
        print(f"{layer.__class__.__name__:25s} -> {tuple(tensor.shape)}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """完成一個 epoch；有 optimizer 時訓練，否則只做 validation。"""

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(
        loader,
        desc="Train" if training else "Valid",
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            # PyTorch 的 gradient 預設會累加，所以每個 batch 都要先清空。
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            # Forward：模型把圖片轉成 [batch, 11] logits。
            logits = model(images)
            # Loss：衡量 logits 和正確類別之間的差距。
            loss = criterion(logits, labels)

            if training:
                # Backward：計算 loss 對每個可訓練參數的 gradient。
                loss.backward()
                # Optimizer：Adam 根據 gradient 更新模型參數。
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def write_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> None:
    """用最佳模型預測 testing data，產生 Kaggle 要求的 CSV。"""

    model.eval()
    predictions: list[int] = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Test"):
            logits = model(images.to(device, non_blocking=True))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Category"])
        writer.writerows(enumerate(predictions))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML2021 HW3 learning version")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="開啟保守的 RandomResizedCrop 與 HorizontalFlip",
    )
    parser.add_argument(
        "--show-shapes",
        action="store_true",
        help="顯示模型每一層的 Tensor shape 後結束",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs 必須大於 0")
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"找不到資料集：{DATA_ROOT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = Classifier().to(device)
    if args.show_shapes:
        show_tensor_shapes(model, device)
        return

    train_loader, valid_loader, test_loader = build_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
    )
    print(
        f"Images: train={len(train_loader.dataset)}, "
        f"valid={len(valid_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )
    print(f"Augmentation: {'on' if args.augment else 'off'}")

    # 多類別單選分類使用 CrossEntropyLoss。
    criterion = nn.CrossEntropyLoss()
    # 先固定教材設定，只觀察一個 optimizer，不急著調參。
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
    )

    checkpoint_path = ROOT / "learning_best_model.pt"
    prediction_path = ROOT / "predict_learning.csv"
    best_valid_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        valid_loss, valid_accuracy = run_epoch(
            model,
            valid_loader,
            criterion,
            device,
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"valid loss={valid_loss:.4f}, acc={valid_accuracy:.4f}"
        )

        # Testing 沒有答案，所以只能用 validation 選模型。
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved: {checkpoint_path.name}")

    # 訓練結束後載入 validation accuracy 最高的模型，而不是最後一輪。
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    write_predictions(model, test_loader, device, prediction_path)
    print(f"Saved: {prediction_path.name}")


if __name__ == "__main__":
    main()
