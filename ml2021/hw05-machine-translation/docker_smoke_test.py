"""驗證 HW5 Docker 內的 GPU、核心套件、資料與一個梯度更新。"""

from __future__ import annotations

import json
from pathlib import Path

import fairseq
import numpy as np
import sentencepiece as spm
import torch


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "DATA" / "rawdata" / "ted2020"


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as file:
        return sum(1 for _ in file)


def main() -> None:
    print("=== 環境 ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"fairseq: {fairseq.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("Container 看不到 CUDA GPU，請先檢查 Docker Desktop 的 WSL2/GPU 設定。")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n=== 資料 ===")
    manifest_path = DATA_ROOT / "preprocess_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到 {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    for split, expected in (("train", 390_041), ("valid", 3_939), ("test", 4_000)):
        en_count = count_lines(DATA_ROOT / f"{split}.en")
        zh_count = count_lines(DATA_ROOT / f"{split}.zh")
        if en_count != zh_count or en_count != expected:
            raise RuntimeError(f"{split} 行數異常：en={en_count}, zh={zh_count}, expected={expected}")

    model_path = DATA_ROOT / "spm8000.model"
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    if tokenizer.get_piece_size() != 8_000:
        raise RuntimeError(f"SentencePiece vocabulary 異常：{tokenizer.get_piece_size()}")
    print("SentencePiece vocab: 8000")

    print("\n=== GPU forward/backward/optimizer ===")
    model = torch.nn.Sequential(
        torch.nn.Linear(40, 80),
        torch.nn.ReLU(),
        torch.nn.Linear(80, 8_000),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(8, 40, device=device)
    targets = torch.randint(0, 8_000, (8,), device=device)
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    print(f"loss: {loss.item():.6f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
