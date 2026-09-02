"""ML2021 HW5 訓練前資料處理。

流程：原始句對 -> 清理與過濾 -> train/valid 切分 -> SentencePiece -> tokenized 檔案。
原始資料只讀取不修改；所有產物都寫回 DATA/rawdata/ted2020。
"""

from __future__ import annotations

import argparse
import importlib.resources
import json
import random
import re
import shutil
import tempfile
from pathlib import Path

import sentencepiece as spm


SEED = 73
VOCAB_SIZE = 8_000
VALID_RATIO = 0.01


def str_q2b(text: str) -> str:
    """將全形 ASCII 字元與全形空格轉成半形。"""
    output: list[str] = []
    for char in text:
        code = ord(char)
        if code == 12288:
            code = 32
        elif 65281 <= code <= 65374:
            code -= 65248
        output.append(chr(code))
    return "".join(output)


def clean_sentence(sentence: str, lang: str) -> str:
    """沿用助教範例的英／中文正規化規則。"""
    if lang == "en":
        sentence = re.sub(r"\([^()]*\)", "", sentence)
        sentence = sentence.replace("-", "")
        sentence = re.sub(r'([.,;!?()\"])', r" \1 ", sentence)
    elif lang == "zh":
        sentence = str_q2b(sentence)
        sentence = re.sub(r"\([^()]*\)", "", sentence)
        sentence = sentence.replace(" ", "")
        sentence = sentence.replace("—", "")
        sentence = sentence.replace("“", '"').replace("”", '"')
        sentence = sentence.replace("_", "")
        sentence = re.sub(r'([。,;!?()\"~「」])', r" \1 ", sentence)
    else:
        raise ValueError(f"不支援的語言：{lang}")
    return " ".join(sentence.strip().split())


def sentence_length(sentence: str, lang: str) -> int:
    return len(sentence) if lang == "zh" else len(sentence.split())


def load_and_clean_pairs(en_path: Path, zh_path: Path) -> tuple[list[str], list[str]]:
    """清理平行語料，並移除空句、過長句及長度比例異常句。"""
    clean_en: list[str] = []
    clean_zh: list[str] = []

    with en_path.open(encoding="utf-8") as en_file, zh_path.open(encoding="utf-8") as zh_file:
        en_lines = en_file.readlines()
        zh_lines = zh_file.readlines()

    if len(en_lines) != len(zh_lines):
        raise ValueError(f"中英文行數不一致：en={len(en_lines)}, zh={len(zh_lines)}")

    for raw_en, raw_zh in zip(en_lines, zh_lines, strict=True):
        en = clean_sentence(raw_en.strip(), "en")
        zh = clean_sentence(raw_zh.strip(), "zh")
        en_len = sentence_length(en, "en")
        zh_len = sentence_length(zh, "zh")

        if en_len < 1 or zh_len < 1 or en_len > 1_000 or zh_len > 1_000:
            continue
        if en_len / zh_len > 9 or zh_len / en_len > 9:
            continue
        clean_en.append(en)
        clean_zh.append(zh)

    return clean_en, clean_zh


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def split_pairs(en_lines: list[str], zh_lines: list[str]) -> dict[str, list[str]]:
    """使用助教程式相同的 seed 與 99/1 規則切分對齊句對。"""
    labels = list(range(len(en_lines)))
    random.Random(SEED).shuffle(labels)
    train_limit = 1.0 - VALID_RATIO

    result = {"train_en": [], "train_zh": [], "valid_en": [], "valid_zh": []}
    for index, (en, zh) in enumerate(zip(en_lines, zh_lines, strict=True)):
        split = "train" if labels[index] / len(labels) < train_limit else "valid"
        result[f"{split}_en"].append(en)
        result[f"{split}_zh"].append(zh)
    return result


def encode_file(processor: spm.SentencePieceProcessor, source: Path, target: Path) -> None:
    with source.open(encoding="utf-8") as input_file, target.open("w", encoding="utf-8", newline="\n") as output_file:
        for line in input_file:
            pieces = processor.encode(line.strip(), out_type=str)
            output_file.write(" ".join(pieces) + "\n")


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as file:
        return sum(1 for _ in file)


def train_sentencepiece(root: Path) -> None:
    """在純 ASCII 暫存路徑訓練，避開 SentencePiece 的 Windows 中文路徑問題。"""
    corpus_names = ("train.clean.en", "valid.clean.en", "train.clean.zh", "valid.clean.zh")
    with tempfile.TemporaryDirectory(prefix="mlhw5_spm_") as temp_name:
        temp_root = Path(temp_name)
        temp_data_dir = temp_root / "package_data"
        package_data = Path(str(importlib.resources.files("sentencepiece"))) / "package_data"
        shutil.copytree(package_data, temp_data_dir)
        spm.SetDataDir(str(temp_data_dir))

        temp_corpora: list[Path] = []
        for name in corpus_names:
            temp_path = temp_root / name
            shutil.copyfile(root / name, temp_path)
            temp_corpora.append(temp_path)

        temp_prefix = temp_root / f"spm{VOCAB_SIZE}"
        spm.SentencePieceTrainer.train(
            input=",".join(str(path) for path in temp_corpora),
            model_prefix=str(temp_prefix),
            vocab_size=VOCAB_SIZE,
            character_coverage=1.0,
            model_type="unigram",
            input_sentence_size=1_000_000,
            shuffle_input_sentence=True,
            normalization_rule_name="nmt_nfkc_cf",
        )
        shutil.copyfile(temp_prefix.with_suffix(".model"), root / f"spm{VOCAB_SIZE}.model")
        shutil.copyfile(temp_prefix.with_suffix(".vocab"), root / f"spm{VOCAB_SIZE}.vocab")


def main() -> None:
    parser = argparse.ArgumentParser(description="準備 ML2021 HW5 訓練資料")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "DATA" / "rawdata" / "ted2020")
    parser.add_argument("--force", action="store_true", help="覆寫既有的前處理產物")
    args = parser.parse_args()
    root = args.data_root.resolve()

    required = [root / "train_dev.raw.en", root / "train_dev.raw.zh", root / "test.raw.en", root / "test.raw.zh"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少原始資料：\n" + "\n".join(missing))

    final_outputs = [root / "train.en", root / "train.zh", root / "valid.en", root / "valid.zh", root / "test.en", root / "test.zh"]
    if not args.force and all(path.is_file() for path in final_outputs) and (root / f"spm{VOCAB_SIZE}.model").is_file():
        print("前處理產物已存在；如需重建請加 --force。")
        return

    print("[1/4] 清理與過濾訓練句對")
    clean_en, clean_zh = load_and_clean_pairs(required[0], required[1])
    write_lines(root / "train_dev.raw.clean.en", clean_en)
    write_lines(root / "train_dev.raw.clean.zh", clean_zh)

    print("[2/4] 以 seed=73 切分 train/valid")
    splits = split_pairs(clean_en, clean_zh)
    for name, lines in splits.items():
        split, lang = name.split("_")
        write_lines(root / f"{split}.clean.{lang}", lines)

    # 測試集不套用長度過濾，只做文字正規化，保留原始順序與 4,000 筆資料。
    for lang in ("en", "zh"):
        source = root / f"test.raw.{lang}"
        lines = [clean_sentence(line.strip(), lang) for line in source.read_text(encoding="utf-8").splitlines()]
        write_lines(root / f"test.raw.clean.{lang}", lines)

    print("[3/4] 訓練共同 SentencePiece unigram 模型（vocab=8000）")
    train_sentencepiece(root)

    print("[4/4] 將 train/valid/test 轉成 subword token")
    # 使用 bytes 載入模型，同樣避免原生函式在 Windows 無法解析中文路徑。
    processor = spm.SentencePieceProcessor(model_proto=(root / f"spm{VOCAB_SIZE}.model").read_bytes())
    source_names = {"train": "train.clean", "valid": "valid.clean", "test": "test.raw.clean"}
    for split, source_prefix in source_names.items():
        for lang in ("en", "zh"):
            encode_file(processor, root / f"{source_prefix}.{lang}", root / f"{split}.{lang}")

    manifest = {
        "seed": SEED,
        "valid_ratio": VALID_RATIO,
        "vocab_size": VOCAB_SIZE,
        "raw_parallel_pairs": count_lines(required[0]),
        "clean_parallel_pairs": len(clean_en),
        "train_pairs": len(splits["train_en"]),
        "valid_pairs": len(splits["valid_en"]),
        "test_pairs": count_lines(root / "test.en"),
    }
    (root / "preprocess_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
