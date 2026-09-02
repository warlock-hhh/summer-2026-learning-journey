from __future__ import annotations

"""
ML2021 HW3 參考高分版。

重現使用者提供 notebook 的主要方法：
1. 五層 CNN（64 -> 128 -> 256 -> 512 -> 1024）。
2. 強 Data Augmentation。
3. Adam(lr=3e-4, weight_decay=1e-5)，最多訓練 500 epochs。
4. Validation accuracy 超過 0.70 後，每 5 epochs 重新產生 pseudo labels。
5. 只收 Softmax confidence 超過 0.90 的 unlabeled 圖片。

針對 RTX 3050 Laptop / Windows 的調整：
- batch size 預設 32（原作者為 128）。
- num_workers 預設 0（原作者為 2）。
- 每個 epoch 保存 latest checkpoint，可用 --resume 接續訓練。
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import DatasetFolder

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "food-11"
BEST_MODEL_PATH = ROOT / "reference_best_model.pt"
LATEST_CHECKPOINT_PATH = ROOT / "reference_latest_checkpoint.pt"
PREDICTION_PATH = ROOT / "predict_reference.csv"
HISTORY_PATH = ROOT / "reference_history.csv"
CURVES_PATH = ROOT / "reference_training_curves.png"


def load_rgb_image(path: str) -> Image.Image:
    """將所有 JPG 統一讀成三個 channel 的 RGB 圖片。"""

    with Image.open(path) as image:
        return image.convert("RGB")


class PseudoLabelDataset(Dataset):
    """保存被 Teacher 選中的圖片路徑及其 pseudo label。"""

    def __init__(
        self,
        samples: list[tuple[str, int]],
        transform,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, pseudo_label = self.samples[index]
        image = load_rgb_image(image_path)
        image = self.transform(image)
        return image, pseudo_label


class Classifier(nn.Module):
    """參考 notebook 的五層 CNN，輸出 Food-11 的 11 個 logits。"""

    def __init__(self) -> None:
        super().__init__()

        # 原作者的 Conv2d 沒有 padding，所以長寬會在卷積時各減少 2。
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 五次卷積與 pooling 後是 [B, 1024, 2, 2]，共 4096 個特徵。
        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 11),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn_layers(images)
        return self.fc_layers(features.flatten(1))


def build_transforms():
    """建立參考 notebook 的 training transform 與乾淨 evaluation transform。"""

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((128, 128)),
            transforms.RandomChoice(
                [
                    transforms.AutoAugment(),
                    transforms.AutoAugment(
                        transforms.AutoAugmentPolicy.CIFAR10
                    ),
                    transforms.AutoAugment(
                        transforms.AutoAugmentPolicy.SVHN
                    ),
                ]
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomAffine(
                degrees=20,
                translate=(0.2, 0.2),
                scale=(0.7, 1.3),
            ),
            transforms.ToTensor(),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    return train_transform, evaluation_transform


def build_datasets():
    """建立 labeled、validation、unlabeled 與 testing datasets。"""

    train_transform, evaluation_transform = build_transforms()
    labeled_set = DatasetFolder(
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
    # 產生 pseudo labels 時使用乾淨圖片，避免信心值被隨機增強干擾。
    # 選中後的圖片會由 PseudoLabelDataset 改用 train_transform。
    unlabeled_set = DatasetFolder(
        DATA_ROOT / "training" / "unlabeled",
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
    return (
        labeled_set,
        valid_set,
        unlabeled_set,
        test_set,
        train_transform,
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def create_pseudo_dataset(
    model: nn.Module,
    unlabeled_set: DatasetFolder,
    train_transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    threshold: float,
) -> PseudoLabelDataset:
    """以目前模型重新掃描 unlabeled data，只保留高信心預測。"""

    loader = make_loader(
        unlabeled_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    model.eval()
    selected_samples: list[tuple[str, int]] = []
    class_counts = [0] * 11
    sample_offset = 0

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Refresh pseudo labels"):
            images = images.to(device, non_blocking=True)
            probabilities = torch.softmax(model(images), dim=1)
            confidences, predictions = probabilities.max(dim=1)

            for batch_index in range(images.size(0)):
                if confidences[batch_index].item() >= threshold:
                    dataset_index = sample_offset + batch_index
                    image_path = unlabeled_set.samples[dataset_index][0]
                    pseudo_label = predictions[batch_index].item()
                    selected_samples.append((image_path, pseudo_label))
                    class_counts[pseudo_label] += 1
            sample_offset += images.size(0)

    print(
        f"Pseudo labels: {len(selected_samples)}/{len(unlabeled_set)} "
        f"(threshold={threshold:.2f})"
    )
    print(
        "Pseudo classes: "
        + " ".join(
            f"c{class_index:02d}={count}"
            for class_index, count in enumerate(class_counts)
        )
    )
    return PseudoLabelDataset(selected_samples, train_transform)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float | None]:
    """執行一個 training 或 validation epoch。"""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    gradient_norm_sum = 0.0
    gradient_steps = 0

    for images, labels in tqdm(
        loader,
        desc="Train" if training else "Valid",
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=10,
                )
                gradient_norm_sum += float(gradient_norm)
                gradient_steps += 1
                optimizer.step()

        current_batch_size = labels.size(0)
        total_loss += loss.item() * current_batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += current_batch_size

    mean_gradient_norm = (
        gradient_norm_sum / gradient_steps if training else None
    )
    return (
        total_loss / total_samples,
        total_correct / total_samples,
        mean_gradient_norm,
    )


def save_history_row(
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    valid_loss: float,
    valid_accuracy: float,
    gradient_norm: float,
    pseudo_count: int,
    resume: bool,
) -> None:
    """將每輪結果追加到 CSV，長時間訓練中斷也不會遺失紀錄。"""

    write_header = not HISTORY_PATH.exists() or (epoch == 1 and not resume)
    mode = "a" if HISTORY_PATH.exists() and not write_header else "w"
    with HISTORY_PATH.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(
                [
                    "Epoch",
                    "TrainLoss",
                    "TrainAccuracy",
                    "ValidLoss",
                    "ValidAccuracy",
                    "GradientNorm",
                    "PseudoCount",
                ]
            )
        writer.writerow(
            [
                epoch,
                f"{train_loss:.8f}",
                f"{train_accuracy:.8f}",
                f"{valid_loss:.8f}",
                f"{valid_accuracy:.8f}",
                f"{gradient_norm:.8f}",
                pseudo_count,
            ]
        )


def save_training_curves() -> None:
    """從 history CSV 重畫 train/valid curves，固定覆蓋同一張 PNG。"""

    with HISTORY_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return

    epochs = [int(row["Epoch"]) for row in rows]
    train_losses = [float(row["TrainLoss"]) for row in rows]
    valid_losses = [float(row["ValidLoss"]) for row in rows]
    train_accs = [float(row["TrainAccuracy"]) for row in rows]
    valid_accs = [float(row["ValidAccuracy"]) for row in rows]
    best_index = max(range(len(valid_accs)), key=valid_accs.__getitem__)
    best_epoch = epochs[best_index]

    image = Image.new("RGB", (1400, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (20, 15),
        f"Reference training curves - best epoch: {best_epoch}",
        fill="black",
    )

    def draw_panel(
        box: tuple[int, int, int, int],
        title: str,
        train_values: list[float],
        valid_values: list[float],
        y_max: float,
    ) -> None:
        left, top, right, bottom = box
        plot_left, plot_top = left + 70, top + 45
        plot_right, plot_bottom = right - 25, bottom - 55
        draw.text((left + 10, top + 10), title, fill="black")

        for index in range(6):
            y = plot_bottom - (plot_bottom - plot_top) * index / 5
            value = y_max * index / 5
            draw.line((plot_left, y, plot_right, y), fill=(220, 220, 220))
            draw.text((left + 10, y - 7), f"{value:.2f}", fill="black")

        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=2)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=2)

        denominator = max(len(epochs) - 1, 1)

        def points(values: list[float]) -> list[tuple[float, float]]:
            return [
                (
                    plot_left + (plot_right - plot_left) * i / denominator,
                    plot_bottom
                    - (plot_bottom - plot_top) * min(value, y_max) / y_max,
                )
                for i, value in enumerate(values)
            ]

        train_points = points(train_values)
        valid_points = points(valid_values)
        if len(train_points) > 1:
            draw.line(train_points, fill=(31, 119, 180), width=3)
            draw.line(valid_points, fill=(255, 127, 14), width=3)
        else:
            draw.ellipse((*train_points[0], *train_points[0]), fill=(31, 119, 180))
            draw.ellipse((*valid_points[0], *valid_points[0]), fill=(255, 127, 14))

        best_x = plot_left + (plot_right - plot_left) * best_index / denominator
        draw.line((best_x, plot_top, best_x, plot_bottom), fill=(214, 39, 40), width=2)
        draw.text((plot_left, plot_bottom + 12), f"Epoch {epochs[0]}", fill="black")
        draw.text((plot_right - 65, plot_bottom + 12), f"Epoch {epochs[-1]}", fill="black")

    loss_max = max(train_losses + valid_losses) * 1.05
    draw_panel((10, 45, 695, 590), "Cross-entropy loss (lower is better)", train_losses, valid_losses, max(loss_max, 0.01))
    draw_panel((705, 45, 1390, 590), "Accuracy (higher is better)", train_accs, valid_accs, 1.0)
    draw.line((90, 570, 125, 570), fill=(31, 119, 180), width=3)
    draw.text((135, 562), "Train", fill="black")
    draw.line((205, 570, 240, 570), fill=(255, 127, 14), width=3)
    draw.text((250, 562), "Valid", fill="black")
    draw.line((320, 570, 355, 570), fill=(214, 39, 40), width=2)
    draw.text((365, 562), "Best epoch", fill="black")
    image.save(CURVES_PATH)


def write_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Test"):
            logits = model(images.to(device, non_blocking=True))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    with PREDICTION_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Category"])
        writer.writerows(enumerate(predictions))
    print(f"Saved: {PREDICTION_PATH.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML2021 HW3 reference recipe")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pseudo-trigger",
        type=float,
        default=0.70,
        help="最佳 validation accuracy 超過此值才啟動 self-labeling",
    )
    parser.add_argument(
        "--pseudo-threshold",
        type=float,
        default=0.90,
        help="收下 pseudo label 所需的最低 Softmax confidence",
    )
    parser.add_argument(
        "--pseudo-refresh",
        type=int,
        default=5,
        help="啟動半監督後，每幾個 epochs 重新產生 pseudo labels",
    )
    parser.add_argument(
        "--no-semi",
        action="store_true",
        help="完全關閉 pseudo-label，只執行 supervised learning",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="從 reference_latest_checkpoint.pt 接續訓練",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"找不到資料集：{DATA_ROOT}")
    if args.epochs <= 0:
        raise ValueError("--epochs 必須大於 0")
    if args.batch_size <= 1:
        raise ValueError("五層模型含 BatchNorm1d，--batch-size 必須大於 1")
    if not 0.0 < args.pseudo_trigger <= 1.0:
        raise ValueError("--pseudo-trigger 必須介於 0 與 1 之間")
    if not 0.0 < args.pseudo_threshold <= 1.0:
        raise ValueError("--pseudo-threshold 必須介於 0 與 1 之間")
    if args.pseudo_refresh <= 0:
        raise ValueError("--pseudo-refresh 必須大於 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    (
        labeled_set,
        valid_set,
        unlabeled_set,
        test_set,
        train_transform,
    ) = build_datasets()
    valid_loader = make_loader(
        valid_set,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )
    test_loader = make_loader(
        test_set,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
    )

    start_epoch = 1
    best_valid_accuracy = 0.0
    if args.resume:
        if not LATEST_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                f"找不到接續訓練檔：{LATEST_CHECKPOINT_PATH}"
            )
        checkpoint = torch.load(
            LATEST_CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_valid_accuracy = float(checkpoint["best_valid_accuracy"])
        print(
            f"Resumed: epoch={start_epoch - 1}, "
            f"best valid acc={best_valid_accuracy:.4f}"
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(
        f"Images: labeled={len(labeled_set)}, valid={len(valid_set)}, "
        f"unlabeled={len(unlabeled_set)}, test={len(test_set)}"
    )
    print(
        f"Recipe: epochs={args.epochs}, batch={args.batch_size}, "
        f"lr=3e-4, weight_decay=1e-5"
    )
    print(
        "Dynamic self-labeling: "
        + (
            f"trigger>{args.pseudo_trigger:.2f}, "
            f"threshold={args.pseudo_threshold:.2f}, "
            f"refresh every {args.pseudo_refresh} epochs"
            if not args.no_semi
            else "off"
        )
    )

    pseudo_set: PseudoLabelDataset | None = None
    for epoch in range(start_epoch, args.epochs + 1):
        semi_ready = (
            not args.no_semi
            and best_valid_accuracy > args.pseudo_trigger
        )
        # 原 notebook 只在固定週期更新 pseudo labels；模型剛超過門檻時，
        # 會等到下一個 refresh epoch 才加入 unlabeled data。
        # resume 後 pseudo_set 尚未存在於記憶體，因此在恢復的第一個 epoch 重建一次。
        should_refresh = semi_ready and (
            epoch % args.pseudo_refresh == 0
            or (args.resume and epoch == start_epoch)
        )
        if should_refresh:
            pseudo_set = create_pseudo_dataset(
                model,
                unlabeled_set,
                train_transform,
                device,
                args.batch_size,
                args.num_workers,
                args.pseudo_threshold,
            )

        training_set: Dataset = labeled_set
        if pseudo_set is not None:
            training_set = ConcatDataset([labeled_set, pseudo_set])
        train_loader = make_loader(
            training_set,
            args.batch_size,
            args.num_workers,
            shuffle=True,
            # 避免最後一批只有一張圖，BatchNorm1d 無法計算 variance。
            drop_last=True,
        )

        train_loss, train_accuracy, gradient_norm = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        valid_loss, valid_accuracy, _ = run_epoch(
            model,
            valid_loader,
            criterion,
            device,
        )
        pseudo_count = len(pseudo_set) if pseudo_set is not None else 0
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"valid loss={valid_loss:.4f}, acc={valid_accuracy:.4f} | "
            f"grad={gradient_norm:.3f} | pseudo={pseudo_count}"
        )

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best: {BEST_MODEL_PATH.name}")

        # latest 保存目前模型與 Adam 狀態，用於真正接續而非重新開始。
        torch.save(
            {
                "epoch": epoch,
                "best_valid_accuracy": best_valid_accuracy,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            LATEST_CHECKPOINT_PATH,
        )
        save_history_row(
            epoch,
            train_loss,
            train_accuracy,
            valid_loss,
            valid_accuracy,
            float(gradient_norm),
            pseudo_count,
            args.resume,
        )
        save_training_curves()

    if not BEST_MODEL_PATH.is_file():
        raise RuntimeError("訓練結束但找不到最佳模型")
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    )
    write_predictions(model, test_loader, device)
    print(f"Saved: {CURVES_PATH.name} (overwritten)")


if __name__ == "__main__":
    main()
