from __future__ import annotations

"""
ML2021 HW3：Food-11 CNN baseline

程式流程：
1. 從 food-11 資料夾建立 training / validation / testing DataLoader。
2. 使用 CNN 將 128×128 RGB 圖片分類成 11 類。
3. 每個 epoch 分別執行訓練與驗證。
4. 保存 validation accuracy 最高的模型參數。
5. 載入最佳模型，對 testing data 產生 predict.csv。
"""

import argparse
import csv
from pathlib import Path

# torch：Tensor、CUDA、模型儲存等核心功能。
import torch
# torch.nn：建立神經網路 layer、loss function。
import torch.nn as nn
# Pillow：讀取 JPG 圖片，也用來輸出訓練曲線（不需額外安裝 matplotlib）。
from PIL import Image, ImageDraw
# DataLoader：把 Dataset 分批、打亂並交給模型。
from torch.utils.data import ConcatDataset, DataLoader, Dataset
# transforms：圖片 Resize、轉 Tensor 與後續的 data augmentation。
from torchvision import transforms
# DatasetFolder：根據子資料夾名稱自動建立 class label。
from torchvision.datasets import DatasetFolder

# tqdm 只負責顯示進度條；即使沒有安裝，訓練仍可執行。
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable


# __file__ 是目前程式檔；parent 取得 ml_hw3 資料夾。
ROOT = Path(__file__).resolve().parent
# 資料集預期放在「程式所在資料夾 / food-11」。
DATA_ROOT = ROOT / "food-11"
# Pseudo-label 分析統一使用這幾個候選信心門檻。
PSEUDO_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def load_rgb_image(path: str) -> Image.Image:
    """開啟 JPG，並統一轉成模型需要的三通道 RGB 圖片。"""
    # with 區塊結束時會自動關閉原始圖片檔案。
    with Image.open(path) as image:
        # convert("RGB") 會建立新圖片，因此原始檔關閉後仍能使用。
        return image.convert("RGB")


class PseudoLabelDataset(Dataset):
    """保存 Teacher 選出的圖片路徑與 pseudo label。"""

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
        # Pseudo-labeled 圖片訓練時使用與 labeled data 相同的 augmentation。
        if self.transform is not None:
            image = self.transform(image)
        return image, pseudo_label


class ResidualBlock(nn.Module):
    """兩層卷積加上一條 shortcut；必要時用 1×1 Conv 對齊形狀。"""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.residual_layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
        )

        # Channel 數或圖片大小改變時，x 無法直接與 residual 相加，
        # 因此用 1×1 Conv 將 shortcut 轉成相同形狀。
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """F(x) + x；shortcut 讓 Gradient 能更直接往前傳。"""
        residual = self.residual_layers(features)
        shortcut = self.shortcut(features)
        return self.relu(residual + shortcut)


class Classifier(nn.Module):
    """輸入 [batch, 3, 128, 128]，輸出 [batch, 11] logits。"""

    def __init__(
        self,
        dropout_rate: float = 0.0,
        architecture: str = "baseline",
    ) -> None:
        # 初始化 nn.Module，讓 PyTorch 能追蹤所有可訓練參數。
        super().__init__()

        if architecture == "baseline":
            # 原始架構完整保留，確保先前 checkpoint 仍可載入。
            self.cnn_layers = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(4),
            )
            self.fc_layers = nn.Sequential(
                # 256×8×8 = 16,384；此層約有 419 萬個參數。
                nn.Linear(256 * 8 * 8, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 11),
            )
        elif architecture == "gap":
            # 增加一層 convolution，再以 Global Average Pooling
            # 取代巨大的 Flatten + Fully Connected 輸入。
            self.cnn_layers = nn.Sequential(
                # [B, 3, 128, 128] -> [B, 64, 64, 64]
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                # -> [B, 128, 32, 32]
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                # -> [B, 256, 16, 16]
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                # -> [B, 512, 8, 8]
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2),
                # [B, 512, 8, 8] -> [B, 512, 1, 1]
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc_layers = nn.Sequential(
                # FC 輸入從原本 16,384 降到 512。
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 11),
            )
        elif architecture == "residual":
            # 小型 Residual CNN：比 GAP 更深，但以較窄的 channel
            # 控制參數量；最後同樣使用 Global Average Pooling。
            self.cnn_layers = nn.Sequential(
                # Stem：[B, 3, 128, 128] -> [B, 32, 64, 64]
                nn.Conv2d(3, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                # 不改變形狀，學習 32-channel 的 residual。
                ResidualBlock(32, 32),
                # stride=2 同時增加 channel 並縮小長寬。
                # -> [B, 64, 32, 32]
                ResidualBlock(32, 64, stride=2),
                # -> [B, 128, 16, 16]
                ResidualBlock(64, 128, stride=2),
                # -> [B, 256, 8, 8]
                ResidualBlock(128, 256, stride=2),
                # -> [B, 256, 1, 1]
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 11),
            )
        else:
            raise ValueError(f"不支援的 architecture：{architecture}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """定義資料通過模型的順序（forward pass）。"""
        # images：[B, 3, 128, 128]
        # features 形狀依架構不同；最後都可用 flatten(1) 轉成 FC 輸入。
        features = self.cnn_layers(images)
        # flatten(1) 保留 batch 維度，把其餘維度攤平成一維。
        # FC layers 最後回傳 [B, 11]。
        return self.fc_layers(features.flatten(1))


def build_loaders(
    batch_size: int,
    num_workers: int,
    augment: bool = False,
) -> tuple[
    DataLoader, DataLoader, DataLoader
]:
    """建立 training、validation、testing 三個 DataLoader。"""

    if augment:
        # 保守的 on-the-fly augmentation：
        # 每次 DataLoader 讀圖時重新抽取隨機參數，但不修改硬碟原圖。
        train_transform = transforms.Compose(
            [
                # 保留原圖 85%～100% 面積，並限制長寬比避免嚴重變形。
                transforms.RandomResizedCrop(
                    128,
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                ),
                # 食物左右朝向不影響類別，因此可用 50% 機率翻轉。
                transforms.RandomHorizontalFlip(p=0.5),
                # 僅小幅旋轉；空白角落以中性灰填補，避免黑角特徵。
                transforms.RandomRotation(10, fill=(128, 128, 128)),
                # 模擬不同拍攝光線，幅度保持保守以免破壞食物顏色。
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.1,
                    hue=0.02,
                ),
                # PIL Image -> [C, H, W]、數值 0.0～1.0 的 Tensor。
                transforms.ToTensor(),
            ]
        )
    else:
        # 無 augmentation 的原始 baseline。
        train_transform = transforms.Compose(
            [
                # 所有圖片統一成 128×128，才能組成相同形狀的 batch。
                transforms.Resize((128, 128)),
                # PIL Image -> torch.Tensor。
                # 數值由 0～255 轉成 0.0～1.0，形狀變成 [C, H, W]。
                transforms.ToTensor(),
            ]
        )

    # validation / testing 不可使用隨機 augmentation，
    # 否則同一張圖片每次評估的內容不同，結果就不穩定。
    eval_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    # DatasetFolder 依照 00～10 子資料夾，自動建立 label 0～10。
    train_set = DatasetFolder(
        DATA_ROOT / "training" / "labeled",
        loader=load_rgb_image,
        # 只讀取副檔名為 .jpg 的檔案。
        extensions=("jpg",),
        transform=train_transform,
    )

    # validation 有正確標籤，只用來評估泛化能力，不更新模型。
    valid_set = DatasetFolder(
        DATA_ROOT / "validation",
        loader=load_rgb_image,
        extensions=("jpg",),
        transform=eval_transform,
    )
    # testing 的 00 資料夾只是 DatasetFolder 所需的目錄形式，
    # 其中的 label 是假的；推論時只使用圖片，不使用該 label。
    test_set = DatasetFolder(
        DATA_ROOT / "testing",
        loader=load_rgb_image,
        extensions=("jpg",),
        transform=eval_transform,
    )

    # 使用 CUDA 時鎖定 CPU memory，可加快 CPU -> GPU 的資料傳輸。
    pin_memory = torch.cuda.is_available()

    # training 必須 shuffle，避免模型一直以固定類別順序看到資料。
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # validation 不可 shuffle；雖然 accuracy 不受順序影響，
    # 固定順序較容易重現與除錯。
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # testing 絕對不可 shuffle，因為 CSV 的 Id 必須對應固定圖片順序。
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # 一次回傳三個 loader，讓 main() 接收。
    return train_loader, valid_loader, test_loader


def build_unlabeled_loader(
    batch_size: int,
    num_workers: int,
) -> tuple[DatasetFolder, DataLoader]:
    """建立 pseudo-label 分析用的 unlabeled Dataset 與 DataLoader。"""

    # Teacher 判斷信心時不可使用隨機 augmentation，
    # 否則同一張圖片每次分析會得到不同的裁切與預測結果。
    eval_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    unlabeled_set = DatasetFolder(
        DATA_ROOT / "training" / "unlabeled",
        loader=load_rgb_image,
        extensions=("jpg",),
        transform=eval_transform,
    )
    unlabeled_loader = DataLoader(
        unlabeled_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return unlabeled_set, unlabeled_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    cutmix: bool = False,
    cutmix_alpha: float = 1.0,
    cutmix_probability: float = 0.5,
) -> tuple[float, float]:
    """
    完整走過一份 DataLoader。

    有傳 optimizer：training mode，會 backward 並更新參數。
    沒傳 optimizer：validation mode，只計算 loss 與 accuracy。
    回傳整個 epoch 的平均 loss 與 accuracy。
    """

    # optimizer 只有訓練時存在，因此可用它判斷目前模式。
    training = optimizer is not None
    # True 等同 model.train()；False 等同 model.eval()。
    # 這會影響 BatchNorm（以及未來可能加入的 Dropout）。
    model.train(training)

    # 累計整個 epoch 的 loss、答對數量與樣本總數。
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    description = "Train" if training else "Valid"
    # DataLoader 每次回傳一個 batch：(images, labels)。
    for images, labels in tqdm(loader, desc=description):
        # 把圖片和標籤移到同一個運算裝置（CUDA 或 CPU）。
        # non_blocking 搭配 pin_memory，可讓 CUDA 傳輸更有效率。
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            # 清除上一個 batch 留下的 gradient。
            # PyTorch 預設會累加 gradient，所以每批更新前必須歸零。
            optimizer.zero_grad()

        # training=True：PyTorch 建立計算圖，以便 backward。
        # training=False：不追蹤 gradient，節省驗證所需記憶體與時間。
        with torch.set_grad_enabled(training):
            # CutMix 只在訓練時隨機啟用：把同一個 batch 的兩張圖片
            # 各取一部分組合成新圖片，標籤也按實際拼接面積加權。
            use_cutmix = (
                training
                and cutmix
                and images.size(0) > 1
                and torch.rand(1).item() < cutmix_probability
            )
            if use_cutmix:
                permutation = torch.randperm(images.size(0), device=device)
                labels_b = labels[permutation]
                # Beta(alpha, alpha) 決定原圖保留比例；alpha=1 為均勻抽樣。
                sampled_lambda = torch.distributions.Beta(
                    cutmix_alpha,
                    cutmix_alpha,
                ).sample().item()
                image_height, image_width = images.shape[-2:]
                cut_ratio = (1.0 - sampled_lambda) ** 0.5
                cut_width = int(image_width * cut_ratio)
                cut_height = int(image_height * cut_ratio)
                center_x = torch.randint(0, image_width, (1,)).item()
                center_y = torch.randint(0, image_height, (1,)).item()
                x1 = max(center_x - cut_width // 2, 0)
                x2 = min(center_x + cut_width // 2, image_width)
                y1 = max(center_y - cut_height // 2, 0)
                y2 = min(center_y + cut_height // 2, image_height)

                images[:, :, y1:y2, x1:x2] = images[
                    permutation, :, y1:y2, x1:x2
                ]
                # 邊界截斷後，必須用真正貼入的面積重新計算 lambda。
                mixed_area = (x2 - x1) * (y2 - y1)
                mix_lambda = 1.0 - mixed_area / (
                    image_width * image_height
                )

            # Forward pass：[B, 3, 128, 128] -> [B, 11] logits。
            logits = model(images)
            if use_cutmix:
                # 一張混合圖片同時對應兩個類別，loss 依面積比例加權。
                loss = (
                    mix_lambda * criterion(logits, labels)
                    + (1.0 - mix_lambda) * criterion(logits, labels_b)
                )
            else:
                # 一般訓練或 validation 使用原始單一標籤。
                loss = criterion(logits, labels)

            if training:
                # Backpropagation：計算 loss 對每個參數的 gradient。
                loss.backward()
                # 限制整體 gradient norm，降低 gradient explosion 風險。
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                # Adam 根據 gradient 更新 Conv、Linear、BatchNorm 參數。
                optimizer.step()

        # 最後一批可能不足 batch_size，因此以實際 label 數量為準。
        batch_count = labels.size(0)
        # loss.item() 是 batch 平均 loss；乘樣本數後才能正確做全體平均。
        total_loss += loss.item() * batch_count
        # CutMix 的 accuracy 也依兩個標籤的圖片面積加權。
        predictions = logits.argmax(dim=1)
        if use_cutmix:
            total_correct += (
                mix_lambda * (predictions == labels).sum().item()
                + (1.0 - mix_lambda)
                * (predictions == labels_b).sum().item()
            )
        else:
            total_correct += (predictions == labels).sum().item()
        total_samples += batch_count

    # 平均 loss = loss 總和 / 圖片數。
    # accuracy = 答對圖片數 / 圖片總數。
    return total_loss / total_samples, total_correct / total_samples


def analyze_pseudo_labels(
    model: nn.Module,
    dataset: DatasetFolder,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> None:
    """分析 Teacher 對 unlabeled data 的信心並覆寫分析 CSV。"""

    model.eval()
    all_confidences: list[float] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Analyze unlabeled"):
            logits = model(images.to(device, non_blocking=True))
            # Softmax 將 11 個 logits 轉成總和為 1 的類別機率。
            probabilities = torch.softmax(logits, dim=1)
            # 每張圖片取最高機率及其類別，作為信心與 pseudo label。
            confidences, predictions = probabilities.max(dim=1)
            all_confidences.extend(confidences.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    if len(all_confidences) != len(dataset.samples):
        raise RuntimeError("Pseudo-label 數量與 unlabeled 圖片數量不一致")

    # 固定檔名，每次分析都覆蓋；保留逐張結果供後續篩選使用。
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Path", "PseudoLabel", "Confidence"])
        for (image_path, _), prediction, confidence in zip(
            dataset.samples,
            all_predictions,
            all_confidences,
        ):
            relative_path = Path(image_path).relative_to(DATA_ROOT)
            writer.writerow(
                [relative_path.as_posix(), prediction, f"{confidence:.8f}"]
            )

    confidences_tensor = torch.tensor(all_confidences)
    predictions_tensor = torch.tensor(all_predictions)
    print(f"Unlabeled images: {len(dataset)}")
    print(
        "Confidence: "
        f"mean={confidences_tensor.mean().item():.4f}, "
        f"median={confidences_tensor.median().item():.4f}, "
        f"max={confidences_tensor.max().item():.4f}"
    )

    # 同時觀察選取總數和各類分布，避免 pseudo labels 嚴重偏向少數類別。
    print("\nPseudo-label distribution")
    print("threshold | total | " + " ".join(f"c{i:02d}" for i in range(11)))
    print("-" * 87)
    for threshold in PSEUDO_THRESHOLDS:
        selected = confidences_tensor >= threshold
        class_counts = torch.bincount(
            predictions_tensor[selected],
            minlength=11,
        )
        counts_text = " ".join(f"{count.item():4d}" for count in class_counts)
        print(
            f"{threshold:9.2f} | "
            f"{selected.sum().item():5d} | "
            f"{counts_text}"
        )

    print(f"\nSaved: {output_path.name} (overwritten)")


def analyze_validation_confidence(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """用有答案的 validation data 檢查 Softmax 信心是否可靠。"""

    model.eval()
    all_confidences: list[float] = []
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Calibrate validation"):
            logits = model(images.to(device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
            all_confidences.extend(confidences.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    confidences_tensor = torch.tensor(all_confidences)
    predictions_tensor = torch.tensor(all_predictions)
    labels_tensor = torch.tensor(all_labels)
    correct = predictions_tensor == labels_tensor

    print("\nValidation confidence calibration")
    print("threshold | selected | correct | actual accuracy")
    print("-" * 52)
    for threshold in PSEUDO_THRESHOLDS:
        selected = confidences_tensor >= threshold
        selected_count = selected.sum().item()
        correct_count = (correct & selected).sum().item()
        accuracy = (
            correct_count / selected_count
            if selected_count > 0
            else 0.0
        )
        print(
            f"{threshold:9.2f} | "
            f"{selected_count:8d} | "
            f"{correct_count:7d} | "
            f"{accuracy:15.2%}"
        )


def create_pseudo_label_dataset(
    model: nn.Module,
    unlabeled_set: DatasetFolder,
    unlabeled_loader: DataLoader,
    train_transform,
    device: torch.device,
    threshold: float,
) -> PseudoLabelDataset:
    """由 Teacher 選出信心達門檻的圖片，建立 pseudo-labeled Dataset。"""

    model.eval()
    selected_samples: list[tuple[str, int]] = []
    class_counts = [0] * 11
    sample_offset = 0

    with torch.no_grad():
        for images, _ in tqdm(unlabeled_loader, desc="Create pseudo labels"):
            logits = model(images.to(device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            # DataLoader 使用 shuffle=False，因此 batch 順序與 samples 完全一致。
            for batch_index in range(images.size(0)):
                if confidences[batch_index].item() >= threshold:
                    dataset_index = sample_offset + batch_index
                    image_path = unlabeled_set.samples[dataset_index][0]
                    pseudo_label = predictions[batch_index].item()
                    selected_samples.append((image_path, pseudo_label))
                    class_counts[pseudo_label] += 1
            sample_offset += images.size(0)

    if not selected_samples:
        raise RuntimeError(
            f"threshold={threshold:.2f} 沒有選到任何 pseudo label"
        )

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


def save_training_curves(
    train_losses: list[float],
    valid_losses: list[float],
    train_accuracies: list[float],
    valid_accuracies: list[float],
    best_epoch: int,
    output_path: Path,
) -> None:
    """將訓練歷史畫成同一張圖；固定路徑會自動覆蓋上一張圖。"""

    # 使用 Pillow 畫圖，避免為了曲線額外安裝 matplotlib。
    width, height = 1400, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (20, 15),
        f"Training curves - best epoch: {best_epoch}",
        fill="black",
    )

    def draw_panel(
        box: tuple[int, int, int, int],
        title: str,
        train_values: list[float],
        valid_values: list[float],
        y_max: float,
    ) -> None:
        """在指定範圍內畫 train/valid 兩條折線。"""

        left, top, right, bottom = box
        plot_left, plot_top = left + 70, top + 45
        plot_right, plot_bottom = right - 25, bottom - 55
        draw.text((left + 10, top + 10), title, fill="black")

        # 畫五等分網格與 Y 軸數值。
        for index in range(6):
            y = plot_bottom - (plot_bottom - plot_top) * index / 5
            value = y_max * index / 5
            draw.line(
                (plot_left, y, plot_right, y),
                fill=(220, 220, 220),
                width=1,
            )
            draw.text((left + 10, y - 7), f"{value:.2f}", fill="black")

        draw.line(
            (plot_left, plot_top, plot_left, plot_bottom),
            fill="black",
            width=2,
        )
        draw.line(
            (plot_left, plot_bottom, plot_right, plot_bottom),
            fill="black",
            width=2,
        )

        epoch_count = len(train_values)

        def to_points(values: list[float]) -> list[tuple[float, float]]:
            denominator = max(epoch_count - 1, 1)
            return [
                (
                    plot_left
                    + (plot_right - plot_left) * index / denominator,
                    plot_bottom
                    - (plot_bottom - plot_top) * value / max(y_max, 1e-12),
                )
                for index, value in enumerate(values)
            ]

        train_points = to_points(train_values)
        valid_points = to_points(valid_values)
        if len(train_points) == 1:
            # 只有一個 epoch 時還沒有線段，因此畫圓點。
            x, y = train_points[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(31, 119, 180))
            x, y = valid_points[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 127, 14))
        else:
            draw.line(train_points, fill=(31, 119, 180), width=3)
            draw.line(valid_points, fill=(255, 127, 14), width=3)

        # 以紅色直線標出 validation accuracy 最佳的 epoch。
        # Pseudo fine-tuning 的 Teacher baseline 記為 epoch 0；
        # 圖上將它畫在最左側，避免超出繪圖範圍。
        plotted_best_epoch = max(best_epoch, 1)
        best_x = plot_left + (
            (plot_right - plot_left)
            * (plotted_best_epoch - 1)
            / max(epoch_count - 1, 1)
        )
        draw.line(
            (best_x, plot_top, best_x, plot_bottom),
            fill=(214, 39, 40),
            width=2,
        )

        # X 軸顯示第一輪、最佳輪與目前最後一輪。
        for epoch_number in sorted({1, plotted_best_epoch, epoch_count}):
            x = plot_left + (
                (plot_right - plot_left)
                * (epoch_number - 1)
                / max(epoch_count - 1, 1)
            )
            draw.text((x - 8, plot_bottom + 10), str(epoch_number), fill="black")

        legend_y = bottom - 25
        draw.line((plot_left, legend_y, plot_left + 28, legend_y), fill=(31, 119, 180), width=3)
        draw.text((plot_left + 35, legend_y - 7), "Train", fill="black")
        draw.line((plot_left + 100, legend_y, plot_left + 128, legend_y), fill=(255, 127, 14), width=3)
        draw.text((plot_left + 135, legend_y - 7), "Valid", fill="black")
        draw.line((plot_left + 200, legend_y, plot_left + 228, legend_y), fill=(214, 39, 40), width=2)
        draw.text((plot_left + 235, legend_y - 7), "Best epoch", fill="black")

    loss_max = max(max(train_losses), max(valid_losses)) * 1.1
    draw_panel(
        (10, 50, 695, 590),
        "Cross-entropy loss (lower is better)",
        train_losses,
        valid_losses,
        loss_max,
    )
    draw_panel(
        (705, 50, 1390, 590),
        "Accuracy (higher is better)",
        train_accuracies,
        valid_accuracies,
        1.0,
    )

    # output_path 固定為 training_curves.png，所以每次都覆蓋而不累積圖片。
    image.save(output_path)


def write_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> None:
    """使用訓練好的模型預測 testing data，並寫成 Kaggle CSV。"""

    # 切換 evaluation mode，讓 BatchNorm 使用訓練期統計值。
    model.eval()
    # 依 testing 圖片順序保存每張圖的預測類別。
    predictions: list[int] = []

    # 推論不需要 backward，因此完全關閉 gradient。
    with torch.no_grad():
        # testing 的 label 是假的，所以用底線表示刻意忽略。
        for images, _ in tqdm(loader, desc="Test"):
            logits = model(images.to(device, non_blocking=True))
            # GPU Tensor -> CPU -> Python list，再加入 predictions。
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    # newline="" 避免 Windows CSV 出現多餘空白行。
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Kaggle 規定的欄位名稱。
        writer.writerow(["Id", "Category"])
        # enumerate 產生 (0, pred0)、(1, pred1)...。
        writer.writerows(enumerate(predictions))


def evaluate_and_write_ensemble(
    gap_model: nn.Module,
    residual_model: nn.Module,
    cutmix_model: nn.Module,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> None:
    """比較三模型權重，選出最佳 validation 組合並輸出 Kaggle CSV。"""

    gap_model.eval()
    residual_model.eval()
    cutmix_model.eval()

    # 只測少量事先決定的比例，避免在 660 張 validation 上過度調參。
    # 權重順序固定為：GAP pseudo、Residual pseudo、Residual CutMix。
    candidate_weights = (
        (0.50, 0.50, 0.00),
        (0.40, 0.40, 0.20),
        (0.30, 0.40, 0.30),
        (0.40, 0.30, 0.30),
        (0.30, 0.30, 0.40),
        (0.25, 0.25, 0.50),
        (0.20, 0.20, 0.60),
    )
    # 每個 batch 保存三個模型對原圖及水平翻轉圖的機率。
    valid_probability_batches: list[tuple[torch.Tensor, ...]] = []
    valid_label_batches: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Ensemble valid"):
            images = images.to(device, non_blocking=True)
            flipped_images = torch.flip(images, dims=(3,))
            valid_probability_batches.append(
                (
                    torch.softmax(gap_model(images), dim=1).cpu(),
                    torch.softmax(residual_model(images), dim=1).cpu(),
                    torch.softmax(cutmix_model(images), dim=1).cpu(),
                    torch.softmax(gap_model(flipped_images), dim=1).cpu(),
                    torch.softmax(
                        residual_model(flipped_images), dim=1
                    ).cpu(),
                    torch.softmax(
                        cutmix_model(flipped_images), dim=1
                    ).cpu(),
                )
            )
            valid_label_batches.append(labels.cpu())

    valid_labels = torch.cat(valid_label_batches)
    validation_results: list[
        tuple[float, float, tuple[float, ...], bool]
    ] = []
    print("\nValidation ensemble + TTA comparison")
    print(" TTA | GAP | Residual | CutMix | loss   | accuracy")
    print("-" * 58)
    for use_tta in (False, True):
        for weights in candidate_weights:
            probability_batches = []
            for probabilities in valid_probability_batches:
                gap_prob, residual_prob, cutmix_prob = probabilities[:3]
                if use_tta:
                    flipped_gap, flipped_residual, flipped_cutmix = (
                        probabilities[3:]
                    )
                    gap_prob = (gap_prob + flipped_gap) / 2
                    residual_prob = (
                        residual_prob + flipped_residual
                    ) / 2
                    cutmix_prob = (cutmix_prob + flipped_cutmix) / 2
                probability_batches.append(
                    weights[0] * gap_prob
                    + weights[1] * residual_prob
                    + weights[2] * cutmix_prob
                )
            ensemble_probabilities = torch.cat(probability_batches)
            predictions = ensemble_probabilities.argmax(dim=1)
            accuracy = (
                (predictions == valid_labels).float().mean().item()
            )
            correct_probabilities = ensemble_probabilities.gather(
                1,
                valid_labels.unsqueeze(1),
            ).squeeze(1)
            loss = -torch.log(
                correct_probabilities.clamp_min(1e-12)
            ).mean().item()
            validation_results.append((accuracy, loss, weights, use_tta))
            print(
                f"{'on ' if use_tta else 'off'} | "
                f"{weights[0]:3.2f} | {weights[1]:8.2f} | "
                f"{weights[2]:6.2f} | {loss:.4f} | {accuracy:.4f}"
            )

    # accuracy 優先；若同分，再選 cross-entropy loss 較低者。
    best_accuracy, best_loss, best_weights, best_tta = max(
        validation_results,
        key=lambda result: (result[0], -result[1]),
    )
    print(
        f"Selected TTA={'on' if best_tta else 'off'} | "
        f"GAP={best_weights[0]:.2f}, "
        f"Residual={best_weights[1]:.2f}, "
        f"CutMix={best_weights[2]:.2f} | "
        f"loss={best_loss:.4f}, acc={best_accuracy:.4f}"
    )

    test_predictions: list[int] = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Ensemble test"):
            images = images.to(device, non_blocking=True)
            gap_probabilities = torch.softmax(gap_model(images), dim=1)
            residual_probabilities = torch.softmax(
                residual_model(images),
                dim=1,
            )
            cutmix_probabilities = torch.softmax(
                cutmix_model(images),
                dim=1,
            )
            if best_tta:
                flipped_images = torch.flip(images, dims=(3,))
                gap_probabilities = (
                    gap_probabilities
                    + torch.softmax(gap_model(flipped_images), dim=1)
                ) / 2
                residual_probabilities = (
                    residual_probabilities
                    + torch.softmax(residual_model(flipped_images), dim=1)
                ) / 2
                cutmix_probabilities = (
                    cutmix_probabilities
                    + torch.softmax(cutmix_model(flipped_images), dim=1)
                ) / 2
            ensemble_probabilities = (
                best_weights[0] * gap_probabilities
                + best_weights[1] * residual_probabilities
                + best_weights[2] * cutmix_probabilities
            )
            test_predictions.extend(
                ensemble_probabilities.argmax(dim=1).cpu().tolist()
            )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Category"])
        writer.writerows(enumerate(test_predictions))
    print(f"Saved: {output_path.name}")


def parse_args() -> argparse.Namespace:
    """讀取使用者從終端機傳入的執行參數。"""

    parser = argparse.ArgumentParser(description="ML2021 HW3 CNN baseline")
    # 完整看過 training set 幾次；預設 1 次用於 smoke test。
    parser.add_argument("--epochs", type=int, default=1)
    # 每次同時送進模型的圖片數；RTX 3050 4 GB 先使用 32。
    parser.add_argument("--batch-size", type=int, default=32)
    # Windows baseline 使用 0，代表由主程序讀圖，最穩定。
    parser.add_argument("--num-workers", type=int, default=0)
    # 開啟保守的 on-the-fly training data augmentation。
    parser.add_argument(
        "--augment",
        action="store_true",
        help="訓練時使用隨機裁切、翻轉、旋轉與輕微調色",
    )
    # Fully connected layers 的 Dropout 比例；0 表示完全關閉。
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="FC hidden layers 的 Dropout 比例，例如 0.3",
    )
    # baseline 保留舊模型；gap 使用更深卷積與 Global Average Pooling。
    parser.add_argument(
        "--architecture",
        choices=("baseline", "gap", "residual"),
        default="baseline",
        help="模型架構：baseline、gap 或 residual（預設：baseline）",
    )
    # 根據 validation loss 自動降低 learning rate。
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="validation loss 停滯時自動降低 learning rate",
    )
    # 額外競賽技巧：在 batch 內剪貼兩張圖片並混合其標籤。
    parser.add_argument(
        "--cutmix",
        action="store_true",
        help="訓練時使用 CutMix（不修改硬碟上的原始圖片）",
    )
    parser.add_argument(
        "--cutmix-alpha",
        type=float,
        default=1.0,
        help="CutMix Beta 分布參數（預設：1.0）",
    )
    parser.add_argument(
        "--cutmix-probability",
        type=float,
        default=0.5,
        help="每個 training batch 使用 CutMix 的機率（預設：0.5）",
    )
    # Adam 的 L2 regularization 強度；預設保留目前最佳實驗的 1e-5。
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Adam 的 weight decay，例如 1e-4（預設：1e-5）",
    )
    # 只分析 unlabeled data，不進行訓練或 testing prediction。
    parser.add_argument(
        "--analyze-pseudo",
        action="store_true",
        help="載入指定實驗的最佳模型，分析 unlabeled pseudo-label 信心",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=None,
        help="分析時指定 Teacher checkpoint；相對路徑以程式資料夾為基準",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="比較 GAP、Residual 與 CutMix 三模型的加權機率",
    )
    # 使用現有最佳模型建立 pseudo labels，再以合併資料繼續訓練。
    parser.add_argument(
        "--pseudo-label",
        action="store_true",
        help="載入 Teacher，加入高信心 unlabeled data 進行 fine-tuning",
    )
    parser.add_argument(
        "--pseudo-threshold",
        type=float,
        default=0.95,
        help="Pseudo label 最低 Softmax 信心（預設：0.95）",
    )
    # 指定此參數時不訓練，只載入 best_model.pt 重新產生 CSV。
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="載入現有 best_model.pt 產生 predict.csv，不重新訓練",
    )
    return parser.parse_args()


def main() -> None:
    """組合資料、模型、訓練、驗證、checkpoint 與推論流程。"""

    args = parse_args()

    # 提早確認資料集存在，避免訓練開始後才出現難懂的路徑錯誤。
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"找不到資料集：{DATA_ROOT}")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout 必須介於 0.0（含）與 1.0（不含）之間")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay 不可小於 0")
    if args.cutmix_alpha <= 0.0:
        raise ValueError("--cutmix-alpha 必須大於 0")
    if not 0.0 <= args.cutmix_probability <= 1.0:
        raise ValueError("--cutmix-probability 必須介於 0 與 1 之間")
    if not 0.0 < args.pseudo_threshold <= 1.0:
        raise ValueError("--pseudo-threshold 必須介於 0 與 1 之間")
    if args.analyze_pseudo and args.pseudo_label:
        raise ValueError("--analyze-pseudo 與 --pseudo-label 不可同時使用")
    if args.teacher_checkpoint is not None and not args.analyze_pseudo:
        raise ValueError("--teacher-checkpoint 目前只能搭配 --analyze-pseudo")
    if args.ensemble and (
        args.analyze_pseudo or args.pseudo_label or args.predict_only
    ):
        raise ValueError(
            "--ensemble 不可與 --analyze-pseudo、"
            "--pseudo-label 或 --predict-only 同時使用"
        )

    # 若 CUDA 可用就使用 GPU，否則退回 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 依命令列的 batch size / workers 建立三個 DataLoader。
    train_loader, valid_loader, test_loader = build_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
    )

    if args.ensemble:
        gap_checkpoint_path = (
            ROOT
            / "best_model_gap_aug_dropout01_scheduler_pseudo090.pt"
        )
        residual_checkpoint_path = (
            ROOT
            / "best_model_residual_aug_dropout01_scheduler_pseudo095.pt"
        )
        cutmix_checkpoint_path = (
            ROOT
            / "best_model_residual_aug_dropout01_scheduler_cutmix.pt"
        )
        for ensemble_checkpoint in (
            gap_checkpoint_path,
            residual_checkpoint_path,
            cutmix_checkpoint_path,
        ):
            if not ensemble_checkpoint.is_file():
                raise FileNotFoundError(
                    f"找不到 Ensemble 模型：{ensemble_checkpoint}"
                )

        gap_model = Classifier(
            dropout_rate=0.1,
            architecture="gap",
        ).to(device)
        residual_model = Classifier(
            dropout_rate=0.1,
            architecture="residual",
        ).to(device)
        cutmix_model = Classifier(
            dropout_rate=0.1,
            architecture="residual",
        ).to(device)
        gap_model.load_state_dict(
            torch.load(
                gap_checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        )
        residual_model.load_state_dict(
            torch.load(
                residual_checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        )
        cutmix_model.load_state_dict(
            torch.load(
                cutmix_checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        )

        ensemble_output_path = (
            ROOT / "predict_ensemble_gap_residual_cutmix_tta.csv"
        )
        print("Ensemble: GAP pseudo + Residual pseudo + Residual CutMix")
        print(f"GAP model: {gap_checkpoint_path.name}")
        print(f"Residual model: {residual_checkpoint_path.name}")
        print(f"CutMix model: {cutmix_checkpoint_path.name}")
        print(
            f"Images: valid={len(valid_loader.dataset)}, "
            f"test={len(test_loader.dataset)}"
        )
        evaluate_and_write_ensemble(
            gap_model,
            residual_model,
            cutmix_model,
            valid_loader,
            test_loader,
            device,
            ensemble_output_path,
        )
        return

    print(f"Augmentation: {'on' if args.augment else 'off'}")
    print(f"Dropout: {args.dropout}")
    print(f"Weight decay: {args.weight_decay:.2e}")
    print(f"Architecture: {args.architecture}")
    print(
        "CutMix: "
        + (
            f"on (alpha={args.cutmix_alpha}, "
            f"probability={args.cutmix_probability})"
            if args.cutmix
            else "off"
        )
    )
    print(
        f"Images: train={len(train_loader.dataset)}, "
        f"valid={len(valid_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    # 建立模型並將全部參數搬到 CUDA 或 CPU。
    model = Classifier(
        dropout_rate=args.dropout,
        architecture=args.architecture,
    ).to(device)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {trainable_parameters:,}")
    # 多類別單選分類使用 CrossEntropyLoss。
    criterion = nn.CrossEntropyLoss()
    # Adam 負責根據 gradient 更新所有可訓練參數。
    optimizer = torch.optim.Adam(
        model.parameters(),
        # learning rate：每次更新參數的步伐大小。
        lr=3e-4,
        # L2 regularization，稍微抑制權重變得過大。
        weight_decay=args.weight_decay,
    )
    # ReduceLROnPlateau 觀察 validation loss：
    # 連續 3 個 epoch 沒有改善後，將 learning rate 乘以 0.5。
    # scheduler 關閉時設為 None，保留原本固定 learning rate 的行為。
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
        if args.scheduler
        else None
    )
    print(f"LR scheduler: {'on' if args.scheduler else 'off'}")

    # 只保存 state_dict（模型學到的參數），不保存 Python 類別本身。
    # 根據實驗設定使用不同檔名，方便公平比較且避免互相覆寫。
    experiment_parts: list[str] = []
    if args.architecture != "baseline":
        experiment_parts.append(args.architecture)
    if args.augment:
        experiment_parts.append("aug")
    if args.dropout > 0:
        # 0.3 -> dropout03，避免檔名包含小數點。
        dropout_tag = str(args.dropout).replace(".", "")
        experiment_parts.append(f"dropout{dropout_tag}")
    if args.scheduler:
        experiment_parts.append("scheduler")
    if args.cutmix:
        experiment_parts.append("cutmix")
    if args.weight_decay != 1e-5:
        # 1e-4 -> wd1em4；只有非預設值才加標籤，以相容舊實驗檔名。
        weight_decay_tag = (
            f"{args.weight_decay:.0e}"
            .replace("e-0", "em")
            .replace("e-", "em")
            .replace("e+0", "e")
            .replace("e+", "e")
        )
        experiment_parts.append(f"wd{weight_decay_tag}")
    # 尚未加入 pseudo 標籤的實驗名稱，就是要載入的 Teacher。
    teacher_experiment_tag = "_".join(experiment_parts)
    teacher_suffix = (
        f"_{teacher_experiment_tag}" if teacher_experiment_tag else ""
    )
    teacher_checkpoint_path = ROOT / f"best_model{teacher_suffix}.pt"
    if args.teacher_checkpoint is not None:
        requested_teacher_path = Path(args.teacher_checkpoint)
        teacher_checkpoint_path = (
            requested_teacher_path
            if requested_teacher_path.is_absolute()
            else ROOT / requested_teacher_path
        )

    if args.pseudo_label:
        # 0.95 -> pseudo095，避免小數點出現在檔名中。
        pseudo_threshold_tag = (
            f"{args.pseudo_threshold:.2f}".replace(".", "")
        )
        experiment_parts.append(f"pseudo{pseudo_threshold_tag}")
    experiment_tag = "_".join(experiment_parts)
    suffix = f"_{experiment_tag}" if experiment_tag else ""
    checkpoint_path = ROOT / f"best_model{suffix}.pt"
    prediction_path = ROOT / f"predict{suffix}.csv"
    # 不依實驗名稱改檔名：每次執行都覆蓋舊圖，避免累積大量圖片。
    curves_path = ROOT / "training_curves.png"
    # Pseudo-label 分析也使用固定檔名，每次執行直接覆蓋。
    pseudo_analysis_path = ROOT / "pseudo_label_analysis.csv"

    # analyze-pseudo：使用現有最佳模型掃描 unlabeled data，不重新訓練。
    if args.analyze_pseudo:
        if not teacher_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"找不到 Teacher 模型：{teacher_checkpoint_path}"
            )
        model.load_state_dict(
            torch.load(
                teacher_checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        )
        unlabeled_set, unlabeled_loader = build_unlabeled_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"Teacher: {teacher_checkpoint_path.name}")
        # Validation 有正確答案，先確認高 Softmax 信心是否真的較準。
        analyze_validation_confidence(model, valid_loader, device)
        # 再將相同 threshold 套用到沒有答案的 unlabeled data。
        analyze_pseudo_labels(
            model,
            unlabeled_set,
            unlabeled_loader,
            device,
            pseudo_analysis_path,
        )
        return

    # predict-only：跳過訓練，直接從既有最佳模型產生提交檔。
    if args.predict_only:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到模型：{checkpoint_path}")
        # map_location 確保模型能載入目前 device。
        # weights_only=True 表示只讀取張量權重，較安全。
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        write_predictions(model, test_loader, device, prediction_path)
        print(f"Saved: {prediction_path.name}")
        return

    if args.pseudo_label:
        if not teacher_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"找不到 Teacher 模型：{teacher_checkpoint_path}"
            )
        # 從目前最佳 Teacher 繼續學，而不是重新隨機初始化。
        model.load_state_dict(
            torch.load(
                teacher_checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        )
        print(f"Teacher: {teacher_checkpoint_path.name}")

        unlabeled_set, unlabeled_loader = build_unlabeled_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        # labeled 與 pseudo-labeled 圖片使用完全相同的 training transform。
        labeled_set = train_loader.dataset
        pseudo_set = create_pseudo_label_dataset(
            model,
            unlabeled_set,
            unlabeled_loader,
            labeled_set.transform,
            device,
            args.pseudo_threshold,
        )
        combined_set = ConcatDataset([labeled_set, pseudo_set])
        train_loader = DataLoader(
            combined_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(
            f"Combined training images: "
            f"{len(labeled_set)} labeled + {len(pseudo_set)} pseudo "
            f"= {len(combined_set)}"
        )

        # Fine-tuning 使用比 Teacher 初始訓練更小的 learning rate。
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = 1e-4
        print("Pseudo fine-tuning LR: 1.00e-04")

        # Epoch 0 先記錄 Teacher 表現；若 fine-tuning 沒改善，仍保留 Teacher。
        teacher_valid_loss, teacher_valid_accuracy = run_epoch(
            model,
            valid_loader,
            criterion,
            device,
        )
        best_accuracy = teacher_valid_accuracy
        best_epoch = 0
        torch.save(model.state_dict(), checkpoint_path)
        print(
            f"Teacher baseline | valid loss={teacher_valid_loss:.4f}, "
            f"acc={teacher_valid_accuracy:.4f}"
        )
        print(f"Saved: {checkpoint_path.name}")
    else:
        # accuracy 不可能低於 0，因此 -1 可保證第一輪一定會保存。
        best_accuracy = -1.0
        best_epoch = 0

    # 保存每個 epoch 的四項指標，提供畫圖使用。
    train_losses: list[float] = []
    valid_losses: list[float] = []
    train_accuracies: list[float] = []
    valid_accuracies: list[float] = []

    # range 的結尾不包含，因此要寫 epochs + 1。
    for epoch in range(1, args.epochs + 1):
        # 記下這個 epoch 實際使用的 learning rate。
        current_lr = optimizer.param_groups[0]["lr"]
        # 傳入 optimizer -> training mode，會更新模型。
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            cutmix=args.cutmix,
            cutmix_alpha=args.cutmix_alpha,
            cutmix_probability=args.cutmix_probability,
        )
        # 不傳 optimizer -> validation mode，不更新模型。
        valid_loss, valid_accuracy = run_epoch(
            model, valid_loader, criterion, device
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"valid loss={valid_loss:.4f}, acc={valid_accuracy:.4f} | "
            f"lr={current_lr:.2e}"
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        # 只在 validation accuracy 創新高時覆寫 checkpoint。
        # 即使後期 overfitting，也能保留泛化表現最好的一輪。
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_epoch = epoch
            # state_dict 包含 Conv、Linear、BatchNorm 的參數與統計值。
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved: {checkpoint_path.name}")

        # 每個 epoch 都更新同一張 PNG；即使訓練被中止，仍保留已完成曲線。
        save_training_curves(
            train_losses,
            valid_losses,
            train_accuracies,
            valid_accuracies,
            best_epoch,
            curves_path,
        )

        if scheduler is not None:
            # 必須在 validation 完成後傳入 valid_loss；
            # 若 loss 長時間沒有下降，下一個 epoch 就會使用較小的 LR。
            scheduler.step(valid_loss)
            next_lr = optimizer.param_groups[0]["lr"]
            if next_lr < current_lr:
                print(
                    f"Learning rate reduced: "
                    f"{current_lr:.2e} -> {next_lr:.2e}"
                )

    print(f"Saved: {curves_path.name} (overwritten)")

    # 訓練結束後，最後一輪不一定最好，因此重新載入最佳 checkpoint。
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    # 使用最佳模型預測 testing data，自動產生 Kaggle 提交檔。
    write_predictions(model, test_loader, device, prediction_path)
    print(f"Saved: {prediction_path.name}")


if __name__ == "__main__":
    # 只有直接執行此檔案時才呼叫 main()；
    # 若其他程式 import 本檔，則只載入函式與類別，不自動開始訓練。
    main()
