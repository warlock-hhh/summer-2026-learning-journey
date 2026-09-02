# -*- coding: utf-8 -*-
"""HW5 Medium Baseline：老師提示的 4-layer Transformer 教學實驗。

本檔案沿用 ``hw05_teacher_baseline.py`` 已驗證的資料、Loss、Optimizer、
training loop、validation、BLEU 與 checkpoint 程式，只把模型核心從：

    雙向 GRU Encoder + Attention + GRU Decoder

替換為：

    Transformer Encoder + Transformer Decoder

老師 HW05 作業講解的 Medium Baseline 提示：

* RNNEncoder -> TransformerEncoder
* RNNDecoder -> TransformerDecoder
* encoder/decoder layers = 4
* FFN dimension = 1024

正式訓練請從 PowerShell 執行 ``train_transformer_medium.ps1``。這條實驗
使用獨立的 ``checkpoints/transformer_medium``，不會覆蓋 GRU/LSTM 成果。
"""

from __future__ import annotations

import argparse
import logging
import shutil
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    base_architecture,
)
from torch.cuda.amp import GradScaler

from hw05_teacher_baseline import (
    LabelSmoothedCrossEntropy,
    NoamOptimizer,
    Seq2Seq,
    get_epoch_iterator,
    load_checkpoint,
    move_sample,
    save_checkpoint,
    set_seed,
    setup_task,
    train_one_epoch,
    validate,
)


LOGGER = logging.getLogger("hw5.transformer-medium")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="老師提示的 4-layer Transformer Medium Baseline"
    )
    parser.add_argument("--data", default="DATA/data-bin/ted2020")
    parser.add_argument("--save-dir", default="checkpoints/transformer_medium")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--max-epoch", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--keep-last-epochs", type=int, default=5)
    parser.add_argument("--resume", default="checkpoint_last.pt")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def transformer_config() -> Namespace:
    """建立老師提示的 Medium Transformer 超參數。

    ``base_architecture`` 只補 fairseq 所需的其他預設欄位；下面明確寫出的
    欄位才是這次實驗要學習與比較的主要設定。
    """
    cfg = Namespace(
        # Token ID 查表後的向量維度，也是 Transformer 的 d_model。
        encoder_embed_dim=256,
        decoder_embed_dim=256,

        # 每層 Attention 後面的 position-wise Feed-Forward Network 維度。
        encoder_ffn_embed_dim=1024,
        decoder_ffn_embed_dim=1024,

        # Encoder/Decoder 各堆疊四層。
        encoder_layers=4,
        decoder_layers=4,

        # 每層 Multi-Head Attention 分成四個 head；每個 head 維度 256/4=64。
        encoder_attention_heads=4,
        decoder_attention_heads=4,

        # Pre-LN：先 LayerNorm 再進 Attention/FFN，深層訓練通常較穩。
        encoder_normalize_before=True,
        decoder_normalize_before=True,

        # ReLU 是老師範例提示使用的 FFN activation。
        activation_fn="relu",
        dropout=0.3,
        attention_dropout=0.1,
        activation_dropout=0.1,

        max_source_positions=1024,
        max_target_positions=1024,
        share_decoder_input_output_embed=True,
    )
    base_architecture(cfg)
    return cfg


def build_transformer_model(task) -> Seq2Seq:
    """明確建立 Transformer Encoder 與 Decoder，再組成 Seq2Seq。"""
    cfg = transformer_config()
    source_dictionary = task.source_dictionary
    target_dictionary = task.target_dictionary

    encoder_embedding = nn.Embedding(
        len(source_dictionary), cfg.encoder_embed_dim, source_dictionary.pad()
    )
    decoder_embedding = nn.Embedding(
        len(target_dictionary), cfg.decoder_embed_dim, target_dictionary.pad()
    )

    # fairseq 的 Transformer build_embedding 會使用 std=d_model^-0.5；若直接
    # 保留 nn.Embedding 預設的 std≈1，又共用 Decoder 輸入/輸出權重，初始
    # logits 會過大，cross entropy 可從合理的 ln(8000)≈8.99 暴增到數百。
    nn.init.normal_(
        encoder_embedding.weight, mean=0.0, std=cfg.encoder_embed_dim ** -0.5
    )
    nn.init.normal_(
        decoder_embedding.weight, mean=0.0, std=cfg.decoder_embed_dim ** -0.5
    )
    nn.init.zeros_(encoder_embedding.weight[source_dictionary.pad()])
    nn.init.zeros_(decoder_embedding.weight[target_dictionary.pad()])

    encoder = TransformerEncoder(cfg, source_dictionary, encoder_embedding)
    decoder = TransformerDecoder(cfg, target_dictionary, decoder_embedding)
    return Seq2Seq(encoder, decoder)


def save_translation_samples(save_dir: Path, epoch: int, stats: dict) -> None:
    """保存 validation 翻譯，讓 BLEU 之外還能人工檢查錯誤。"""
    output_path = save_dir / f"validation_samples_epoch{epoch}.tsv"
    with output_path.open("w", encoding="utf-8", newline="") as output:
        output.write("source\thypothesis\treference\n")
        for source, hypothesis, reference in zip(
            stats["sources"], stats["hypotheses"], stats["references"]
        ):
            output.write(
                f"{source.replace(chr(9), ' ')}\t"
                f"{hypothesis.replace(chr(9), ' ')}\t"
                f"{reference.replace(chr(9), ' ')}\n"
            )


def run_smoke_test(model, task, criterion, optimizer, scaler, device, args) -> None:
    """跑一個真實 batch 的 forward/loss/backward，確認 4 GB GPU 可承受。"""
    iterator = get_epoch_iterator(
        task, "train", 1, args.max_tokens, 0, args.seed
    )
    sample = next(iterator.next_epoch_itr(shuffle=False))
    sample = move_sample(sample, device)
    model.train()
    optimizer.optimizer.zero_grad()

    # 直接沿用已驗證的 GRU 教學檔 loss 資料流，但模型物件是 Transformer。
    from hw05_teacher_baseline import compute_loss

    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        loss = compute_loss(model, criterion, sample) / sample["ntokens"]
    scaler.scale(loss).backward()
    LOGGER.info(
        "Transformer smoke test 成功：batch tokens=%d, token loss=%.4f",
        sample["ntokens"],
        loss.item(),
    )


def generation_config(beam: int) -> Namespace:
    """Beam search 設定；beam=5 表示每一步保留五條候選前綴。"""
    return Namespace(
        beam=beam,
        max_len_a=1.2,
        max_len_b=10,
        min_len=1,
        lenpen=1.0,
        unkpen=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        sampling=False,
        sampling_topk=-1,
        sampling_topp=-1.0,
        diverse_beam_groups=-1,
        diverse_beam_strength=0.5,
        diversity_rate=-1.0,
        constraints=None,
        prefix_allowed_tokens_fn=None,
        print_alignment=False,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    if device.type != "cuda" and not args.cpu:
        raise RuntimeError("找不到 CUDA；若只做除錯，請明確加上 --cpu")

    task = setup_task(args)
    model = build_transformer_model(task).to(device)
    criterion = LabelSmoothedCrossEntropy(
        smoothing=0.1, padding_index=task.target_dictionary.pad()
    ).to(device)
    optimizer = NoamOptimizer(model.parameters(), model_size=256, factor=2.0)
    scaler = GradScaler(enabled=device.type == "cuda")

    LOGGER.info("device=%s", device)
    LOGGER.info("model=%s", model)
    LOGGER.info("trainable parameters=%s", f"{sum(p.numel() for p in model.parameters()):,}")
    LOGGER.info(
        "max_tokens=%d accum_steps=%d effective_tokens≈%d",
        args.max_tokens,
        args.accum_steps,
        args.max_tokens * args.accum_steps,
    )

    if args.smoke_test:
        run_smoke_test(model, task, criterion, optimizer, scaler, device, args)
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best_bleu = 1, float("-inf")
    if not args.no_resume:
        start_epoch, best_bleu = load_checkpoint(
            save_dir / args.resume, model, optimizer, device
        )

    generator = task.build_generator([model], generation_config(args.beam))
    for epoch in range(start_epoch, args.max_epoch + 1):
        train_loss = train_one_epoch(
            model, task, criterion, optimizer, scaler, device, args, epoch
        )
        stats = validate(model, task, criterion, generator, device, args)
        LOGGER.info(
            "epoch=%d train_loss=%.4f valid_loss=%.4f BLEU_zh=%.2f",
            epoch,
            train_loss,
            stats["loss"],
            stats["bleu"],
        )

        save_translation_samples(save_dir, epoch, stats)
        is_best = stats["bleu"] > best_bleu
        best_bleu = max(best_bleu, stats["bleu"])
        epoch_path = save_dir / f"checkpoint{epoch}.pt"
        save_checkpoint(epoch_path, model, optimizer, epoch, stats, best_bleu)
        shutil.copy2(epoch_path, save_dir / "checkpoint_last.pt")
        if is_best:
            shutil.copy2(epoch_path, save_dir / "checkpoint_best.pt")

        old_epoch = epoch - args.keep_last_epochs
        old_checkpoint = save_dir / f"checkpoint{old_epoch}.pt"
        old_samples = save_dir / f"validation_samples_epoch{old_epoch}.tsv"
        if old_checkpoint.exists():
            old_checkpoint.unlink()
        if old_samples.exists():
            old_samples.unlink()


if __name__ == "__main__":
    main()
