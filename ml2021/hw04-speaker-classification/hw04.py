"""ML2021 Spring HW4：以 Self-Attention 進行語者分類的 baseline。

題目：輸入一段語音的 Mel-spectrogram，從 600 位候選人中判斷語者身分。

資料流（B=batch、T=time、F=feature）：
    .pt 特徵 (T, 40) -> 裁切／padding (B, T, 40)
    -> Linear 40→80 -> TransformerEncoderLayer -> mean pooling
    -> 600 類 logits

命令：check-data（檢查資料）、train（訓練）、infer（產生 Kaggle CSV）。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
# 官方壓縮檔解壓後會形成 Dataset/Dataset 兩層目錄。
DEFAULT_DATA_DIR = SCRIPT_DIR / "Dataset" / "Dataset"


def seed_everything(seed: int) -> None:
    """固定亂數來源，使資料切分與實驗結果較容易重現。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_tensor(path: Path) -> torch.Tensor:
    """將課程提供的 .pt Mel-spectrogram 載入 CPU 記憶體。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location="cpu")


class SpeakerDataset(Dataset):
    """訓練資料集：每筆回傳 (Mel 特徵, 語者數字類別)。"""

    def __init__(self, data_dir: Path, segment_len: int = 128) -> None:
        self.data_dir = data_dir
        self.segment_len = segment_len

        # mapping.json：語者文字 ID 與模型數字類別 0~599 的雙向對照。
        mapping = json.loads((data_dir / "mapping.json").read_text(encoding="utf-8"))
        # metadata.json：每位語者擁有哪些 uttr-*.pt 訓練特徵檔。
        metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))["speakers"]
        self.speaker2id = mapping["speaker2id"]
        self.speaker_num = len(metadata)
        # 攤平成 [(特徵檔路徑, 數字 label), ...]，供 __getitem__ 依索引讀取。
        self.data = [
            (utterance["feature_path"], self.speaker2id[speaker])
            for speaker, utterances in metadata.items()
            for utterance in utterances
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        feature_path, speaker = self.data[index]
        # mel shape：(原始 frame 數 T, 40 個 Mel 頻帶)。
        mel = load_tensor(self.data_dir / feature_path).float()
        # 長語音每次隨機取連續 128 frames：降低 Attention 計算量，
        # 也使同一段語音在不同 epoch 取到不同片段，形成簡單資料增強。
        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)
            mel = mel[start : start + self.segment_len]
        return mel, speaker


def collate_batch(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """將不同時間長度的語音補齊，組成 GPU 可平行處理的 batch。"""
    mels, speakers = zip(*batch)
    # 使用 -20（log-Mel 中接近極小能量）補到此 batch 的最大長度。
    # 輸出 mels：(B, T_max, 40)；labels：(B,) 且 dtype 為 long。
    mels = pad_sequence(mels, batch_first=True, padding_value=-20.0)
    return mels, torch.tensor(speakers, dtype=torch.long)


def get_dataloaders(
    data_dir: Path,
    batch_size: int,
    workers: int,
    segment_len: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, int]:
    """固定切出 90% 訓練集、10% 驗證集並建立 DataLoader。"""
    dataset = SpeakerDataset(data_dir, segment_len)
    train_len = int(0.9 * len(dataset))
    generator = torch.Generator().manual_seed(seed)
    train_set, valid_set = random_split(
        dataset, [train_len, len(dataset) - train_len], generator=generator
    )
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        # CUDA 訓練時，pin_memory 可加速 CPU→GPU 的資料傳輸。
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_batch,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    valid_loader = DataLoader(valid_set, shuffle=False, drop_last=False, **common)
    return train_loader, valid_loader, dataset.speaker_num


class Classifier(nn.Module):
    """可調整 Encoder 層數與 head 數的 Transformer 語者分類器。"""

    def __init__(
        self,
        d_model: int = 80,
        n_spks: int = 600,
        dropout: float = 0.1,
        nhead: int = 2,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        pooling: str = "mean",
        pooling_hidden: int = 64,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} 必須能被 nhead={nhead} 整除")
        if num_layers < 1:
            raise ValueError("num_layers 必須至少為 1")
        if pooling not in {"mean", "attention"}:
            raise ValueError("pooling 必須是 mean 或 attention")
        if pooling_hidden < 1:
            raise ValueError("pooling_hidden 必須至少為 1")

        self.num_layers = num_layers
        self.pooling = pooling
        # 原始每個 frame 有 40 個 Mel 特徵；先投影為 80 維內部表示。
        # (B, T, 40) -> (B, T, 80)
        self.prenet = nn.Linear(40, d_model)
        # TransformerEncoderLayer 內部已包含：
        # 1. 2-head Multi-Head Self-Attention（80 / 2 = 每個 head 40 維）
        # 2. Residual connection + LayerNorm
        # 3. Feed-Forward Network（80 -> 256 -> 80）
        # 4. 第二組 Residual connection + LayerNorm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        # num_layers=1 保留原 baseline 的參數名稱，讓舊 model.ckpt 仍可載入；
        # num_layers>1 才使用 TransformerEncoder 疊加多層 Encoder Layer。
        if num_layers == 1:
            self.encoder_layer: nn.TransformerEncoderLayer | None = encoder_layer
            self.encoder: nn.TransformerEncoder | None = None
        else:
            self.encoder_layer = None
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Mean Pooling 將所有 frame 等權平均；Attention Pooling 則學習每個
        # frame 的重要性，再以 softmax 權重做加權平均。
        self.attention_pool = (
            nn.Sequential(
                nn.Linear(d_model, pooling_hidden),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(pooling_hidden, 1),
            )
            if pooling == "attention"
            else None
        )
        # 將整段語音的 80 維摘要轉成 600 位語者的 logits。
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """執行前向傳播，回傳尚未做 Softmax 的分類 logits。"""
        # mels：(B, T, 40)
        hidden = self.prenet(mels)  # (B, T, 80)
        if self.encoder is not None:
            encoded = self.encoder(hidden)
        else:
            assert self.encoder_layer is not None
            encoded = self.encoder_layer(hidden)
        if self.attention_pool is None:
            # Mean Pooling：所有時間 frames 使用相同權重。
            pooled = encoded.mean(dim=1)  # (B, 80)
        else:
            # Attention Pooling：(B,T,80) -> scores/weights (B,T,1) -> (B,80)
            scores = self.attention_pool(encoded)
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(encoded * weights, dim=1)
        return self.pred_layer(pooled)  # (B, n_spks=600)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    training_steps: int,
    cycles: float = 0.5,
) -> LambdaLR:
    """前期線性 warmup，之後以 cosine 曲線逐漸降低 learning rate。"""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, training_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


def get_device(require_gpu: bool) -> torch.device:
    """選擇 CUDA；預設不允許在不知情下退回慢速 CPU 訓練。"""
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required but unavailable. Install a CUDA-enabled PyTorch build "
            "and verify the NVIDIA driver."
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_config(args: argparse.Namespace) -> dict[str, int | float | str]:
    """從命令列參數整理可重建模型的架構設定。"""
    return {
        "d_model": args.d_model,
        "dropout": args.dropout,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "pooling": args.pooling,
        "pooling_hidden": args.pooling_hidden,
    }


def validate(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """不計算梯度地跑驗證集，回傳平均 loss 與 accuracy。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    limit = len(loader) if max_batches is None else min(len(loader), max_batches)
    # 驗證階段不建立反向傳播計算圖，可節省顯存與時間。
    with torch.inference_mode():
        for batch_index, (mels, labels) in enumerate(
            tqdm(loader, total=limit, desc="Valid", unit="batch", ncols=0)
        ):
            if batch_index >= limit:
                break
            mels = mels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(mels)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
    model.train()
    return total_loss / total_samples, total_correct / total_samples


def save_training_history(history: list[dict[str, float | int | None]], path: Path) -> None:
    """將每個 epoch 的訓練／驗證指標寫成 CSV，方便後續比較實驗。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "valid_loss",
        "valid_accuracy",
        "learning_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_training_plots(
    history: list[dict[str, float | int | None]],
    loss_path: Path,
    accuracy_path: Path,
) -> None:
    """將目前所有 epochs 的 loss 與 accuracy 各畫成一張 PNG。"""
    # 使用非互動式後端，讓 PowerShell／背景訓練不需開啟繪圖視窗。
    matplotlib_config_dir = SCRIPT_DIR / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    valid_rows = [row for row in history if row["valid_loss"] is not None]

    loss_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    if valid_rows:
        axis.plot(
            [row["epoch"] for row in valid_rows],
            [row["valid_loss"] for row in valid_rows],
            marker="o",
            label="Validation loss",
        )
    axis.set(title="Loss by epoch", xlabel="Epoch", ylabel="Cross-entropy loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(loss_path, dpi=160)
    plt.close(figure)

    accuracy_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        epochs,
        [row["train_accuracy"] for row in history],
        marker="o",
        label="Train accuracy",
    )
    if valid_rows:
        axis.plot(
            [row["epoch"] for row in valid_rows],
            [row["valid_accuracy"] for row in valid_rows],
            marker="o",
            label="Validation accuracy",
        )
    axis.set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy", ylim=(0.0, 1.0))
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(accuracy_path, dpi=160)
    plt.close(figure)


def train(args: argparse.Namespace) -> None:
    """以 epoch 為單位訓練，並保存驗證正確率最高的模型參數。"""
    seed_everything(args.seed)
    device = get_device(args.require_gpu)
    print(f"[Info] device={device}")
    if device.type == "cuda":
        print(f"[Info] gpu={torch.cuda.get_device_name(0)}")

    train_loader, valid_loader, speaker_num = get_dataloaders(
        args.data_dir, args.batch_size, args.workers, args.segment_len, args.seed
    )
    print(
        f"[Info] speakers={speaker_num}, train={len(train_loader.dataset)}, "
        f"valid={len(valid_loader.dataset)}"
    )

    model_config = get_model_config(args)
    model = Classifier(n_spks=speaker_num, **model_config).to(device)
    # CrossEntropyLoss 直接接收 logits (B, 600) 和整數 labels (B,)；
    # 內部已包含 LogSoftmax，因此 forward 不需要自行做 Softmax。
    criterion = nn.CrossEntropyLoss()
    # AdamW 會根據梯度更新模型參數，並處理 weight decay。
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Scheduler 內部仍需以每個 batch（step）調整 learning rate，
    # 但使用者只需設定 epochs；程式會自動換算總 steps 與 warmup steps。
    batches_per_epoch = len(train_loader)
    if args.max_train_batches is not None:
        batches_per_epoch = min(batches_per_epoch, args.max_train_batches)
    total_training_steps = args.epochs * batches_per_epoch
    warmup_steps = round(args.warmup_epochs * batches_per_epoch)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_training_steps
    )
    best_accuracy = -1.0
    global_step = 0
    history: list[dict[str, float | int | None]] = []
    print(
        f"[Info] epochs={args.epochs}, batches_per_epoch={batches_per_epoch}, "
        f"warmup_epochs={args.warmup_epochs}, total_steps={total_training_steps}"
    )
    print(
        f"[Info] model: layers={args.num_layers}, heads={args.nhead}, "
        f"d_model={args.d_model}, segment_len={args.segment_len}, pooling={args.pooling}"
    )
    # 每個實驗保存完整設定，之後可確認分數究竟來自哪組參數。
    experiment_config = {
        "model": model_config,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "segment_len": args.segment_len,
            "learning_rate": args.lr,
            "warmup_epochs": args.warmup_epochs,
            "seed": args.seed,
        },
    }
    args.config_path.parent.mkdir(parents=True, exist_ok=True)
    args.config_path.write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0
        progress = tqdm(
            train_loader,
            total=batches_per_epoch,
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
            ncols=0,
        )

        for batch_index, (mels, labels) in enumerate(progress):
            if batch_index >= batches_per_epoch:
                break

            # 將輸入與正確答案搬到 GPU。
            mels = mels.to(device, non_blocking=True)  # (B, T, 40)
            labels = labels.to(device, non_blocking=True)

            # 一個 batch 的標準更新：清梯度 -> forward -> loss -> backward -> 更新。
            optimizer.zero_grad(set_to_none=True)
            logits = model(mels)  # (B, 600)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_samples += batch_size
            progress.set_postfix(
                loss=f"{train_loss_sum / train_samples:.3f}",
                acc=f"{train_correct / train_samples:.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        train_loss = train_loss_sum / train_samples
        train_accuracy = train_correct / train_samples
        print(
            f"[Train] epoch={epoch}/{args.epochs}, loss={train_loss:.4f}, "
            f"accuracy={train_accuracy:.4f}, lr={scheduler.get_last_lr()[0]:.6g}"
        )

        # 使用熟悉的 epoch 節奏：每 N 個 epochs 跑一次完整 validation。
        should_validate = epoch % args.validate_every == 0 or epoch == args.epochs
        valid_loss: float | None = None
        valid_accuracy: float | None = None
        if should_validate:
            valid_loss, valid_accuracy = validate(
                valid_loader, model, criterion, device, args.max_valid_batches
            )
            print(
                f"[Valid] epoch={epoch}/{args.epochs}, loss={valid_loss:.4f}, "
                f"accuracy={valid_accuracy:.4f}"
            )

            # 只要本 epoch 創下最佳 validation accuracy，就立即覆寫 model.ckpt。
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                args.save_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "model_state": copy.deepcopy(model.state_dict()),
                    "model_config": model_config,
                    "epoch": epoch,
                    "best_accuracy": best_accuracy,
                    "segment_len": args.segment_len,
                }
                torch.save(checkpoint, args.save_path)
                print(
                    f"[Info] best model saved={args.save_path}, "
                    f"epoch={epoch}, best_accuracy={best_accuracy:.4f}"
                )

        # 每個 epoch 都保存數值與圖表；即使訓練中斷，已完成的紀錄仍在磁碟上。
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_accuracy,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        save_training_history(history, args.history_path)
        save_training_plots(history, args.loss_plot, args.accuracy_plot)
        print(
            f"[Info] history={args.history_path}, loss_plot={args.loss_plot}, "
            f"accuracy_plot={args.accuracy_plot}"
        )


class InferenceDataset(Dataset):
    """Kaggle 測試集：只有特徵檔路徑，沒有正確 speaker label。"""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        # testdata.json 列出 6,000 個等待模型預測的 uttr-*.pt。
        self.data = json.loads((data_dir / "testdata.json").read_text(encoding="utf-8"))[
            "utterances"
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        feature_path = self.data[index]["feature_path"]
        return feature_path, load_tensor(self.data_dir / feature_path).float()


def inference_collate_batch(batch: list[tuple[str, torch.Tensor]]) -> tuple[list[str], torch.Tensor]:
    """組成推論 batch，並保留檔名以便把預測逐列寫回 CSV。"""
    paths, mels = zip(*batch)
    return list(paths), pad_sequence(mels, batch_first=True, padding_value=-20.0)


def infer(args: argparse.Namespace) -> None:
    """載入最佳 checkpoint，預測測試資料並輸出 Kaggle CSV。"""
    device = get_device(args.require_gpu)
    mapping = json.loads((args.data_dir / "mapping.json").read_text(encoding="utf-8"))
    loaded = torch.load(args.model_path, map_location=device, weights_only=True)

    # 新 checkpoint 同時保存架構設定；舊 baseline checkpoint 則只有 state_dict。
    if isinstance(loaded, dict) and "model_state" in loaded:
        model_config = loaded["model_config"]
        state = loaded["model_state"]
        print(
            f"[Info] checkpoint model: layers={model_config['num_layers']}, "
            f"heads={model_config['nhead']}, d_model={model_config['d_model']}, "
            f"pooling={model_config.get('pooling', 'mean')}"
        )
    else:
        model_config = {
            "d_model": 80,
            "dropout": 0.1,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 256,
        }
        state = loaded
        print("[Info] legacy baseline checkpoint detected")

    model = Classifier(n_spks=len(mapping["id2speaker"]), **model_config).to(device)
    model.load_state_dict(state)
    model.eval()

    dataset = InferenceDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=inference_collate_batch,
    )
    results = [["Id", "Category"]]
    with torch.inference_mode():
        for paths, mels in tqdm(loader, desc="Infer", unit="batch", ncols=0):
            # argmax 選出 600 個 logits 中最大的數字類別，再搬回 CPU 寫檔。
            predictions = model(mels.to(device, non_blocking=True)).argmax(dim=1).cpu().tolist()
            results.extend(
                # Kaggle 要文字 speaker ID，因此使用 mapping.json 的 id2speaker 反查。
                [path, mapping["id2speaker"][str(prediction)]]
                for path, prediction in zip(paths, predictions)
            )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(results)
    print(f"[Info] predictions={len(results) - 1}, output={args.output_path}")


def check_data(data_dir: Path) -> None:
    """快速確認必要 JSON 與訓練／測試 .pt 能否正常讀取。"""
    required = ["mapping.json", "metadata.json", "testdata.json"]
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")
    dataset = SpeakerDataset(data_dir)
    test_dataset = InferenceDataset(data_dir)
    sample_mel, sample_label = dataset[0]
    test_path, test_mel = test_dataset[0]
    print(
        f"[Data] train={len(dataset)}, test={len(test_dataset)}, speakers={dataset.speaker_num}, "
        f"train_sample={tuple(sample_mel.shape)}, label={sample_label}, "
        f"test_sample={test_path}:{tuple(test_mel.shape)}"
    )


def build_parser() -> argparse.ArgumentParser:
    """定義 check-data、train、infer 三組命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check-data", help="Validate data and load samples")
    check_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    train_parser = subparsers.add_parser("train", help="Train the baseline model")
    train_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train_parser.add_argument(
        "--run-dir",
        type=Path,
        help="將 checkpoint、曲線與紀錄集中到此實驗資料夾",
    )
    train_parser.add_argument("--save-path", type=Path, default=SCRIPT_DIR / "model.ckpt")
    train_parser.add_argument(
        "--history-path", type=Path, default=SCRIPT_DIR / "training_history.csv"
    )
    train_parser.add_argument(
        "--loss-plot", type=Path, default=SCRIPT_DIR / "loss_curve.png"
    )
    train_parser.add_argument(
        "--accuracy-plot", type=Path, default=SCRIPT_DIR / "accuracy_curve.png"
    )
    train_parser.add_argument(
        "--config-path", type=Path, default=SCRIPT_DIR / "experiment_config.json"
    )
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument(
        "--infer-after-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="訓練完成後，以最佳 checkpoint 自動產生 Kaggle output.csv（預設啟用）",
    )
    train_parser.add_argument(
        "--infer-batch-size",
        type=int,
        default=1,
        help="訓練後自動推論使用的 batch size",
    )
    train_parser.add_argument("--segment-len", type=int, default=128)
    train_parser.add_argument("--num-layers", type=int, default=1)
    train_parser.add_argument("--nhead", type=int, default=2)
    train_parser.add_argument("--d-model", type=int, default=80)
    train_parser.add_argument("--dim-feedforward", type=int, default=256)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument(
        "--pooling",
        choices=("mean", "attention"),
        default="attention",
        help="將 frame 序列濃縮成語音摘要的方法（新訓練預設 attention）",
    )
    train_parser.add_argument(
        "--pooling-hidden",
        type=int,
        default=64,
        help="Attention Pooling 打分網路的隱藏維度",
    )
    # 使用者以熟悉的 epoch 控制訓練；18 epochs 約等於官方 70,000 steps。
    train_parser.add_argument("--epochs", type=int, default=18)
    train_parser.add_argument("--validate-every", type=int, default=1)
    # 官方 warmup 1,000 steps 約為 0.25 epoch。
    train_parser.add_argument("--warmup-epochs", type=float, default=0.25)
    # 僅供 smoke test；正式訓練不設定，便會跑完整 epoch。
    train_parser.add_argument("--max-train-batches", type=int)
    train_parser.add_argument("--max-valid-batches", type=int)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--seed", type=int, default=87)
    train_parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)

    infer_parser = subparsers.add_parser("infer", help="Create Kaggle output CSV")
    infer_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    infer_parser.add_argument(
        "--run-dir", type=Path, help="從此實驗資料夾讀取 model.ckpt 並輸出 output.csv"
    )
    infer_parser.add_argument("--model-path", type=Path, default=SCRIPT_DIR / "model.ckpt")
    infer_parser.add_argument("--output-path", type=Path, default=SCRIPT_DIR / "output.csv")
    infer_parser.add_argument("--batch-size", type=int, default=1)
    infer_parser.add_argument("--workers", type=int, default=0)
    infer_parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.data_dir = args.data_dir.resolve()
    if getattr(args, "run_dir", None) is not None:
        run_dir = args.run_dir.resolve()
        if args.command == "train":
            args.save_path = run_dir / "model.ckpt"
            args.history_path = run_dir / "training_history.csv"
            args.loss_plot = run_dir / "loss_curve.png"
            args.accuracy_plot = run_dir / "accuracy_curve.png"
            args.config_path = run_dir / "experiment_config.json"
        elif args.command == "infer":
            args.model_path = run_dir / "model.ckpt"
            args.output_path = run_dir / "output.csv"
    if args.command == "check-data":
        check_data(args.data_dir)
    elif args.command == "train":
        train(args)
        if args.infer_after_train:
            output_path = (
                args.run_dir.resolve() / "output.csv"
                if args.run_dir is not None
                else SCRIPT_DIR / "output.csv"
            )
            print("[Info] training finished; creating Kaggle output.csv from best checkpoint")
            infer(
                argparse.Namespace(
                    data_dir=args.data_dir,
                    model_path=args.save_path,
                    output_path=output_path,
                    batch_size=args.infer_batch_size,
                    workers=args.workers,
                    require_gpu=args.require_gpu,
                )
            )
    else:
        infer(args)


if __name__ == "__main__":
    # Windows DataLoader 使用多程序 worker 時，必須放在此保護區塊中。
    main()
