# -*- coding: utf-8 -*-
"""HW5 老師架構的可執行教學版 Seq2Seq baseline。

這份檔案的目的不是把模型藏在 ``fairseq-train --arch lstm`` 後面，而是讓
Encoder、Attention、Decoder、Loss、Backpropagation 與驗證流程都能直接閱讀。

保留自老師 ``範例/hw05_zh.py`` 的核心設計：

    Token ID -> Embedding -> 雙向 GRU Encoder
             -> Attention -> 單向 GRU Decoder -> Vocabulary logits

fairseq 在這裡只負責字典、binary dataset、batch iterator 與 beam search；模型核心
由本檔案定義。請注意：老師的 simple baseline 實際使用 GRU，不是 LSTM。

在 Docker 中做最小測試：

    docker compose run --rm hw5 python hw05_teacher_baseline.py --smoke-test

正式訓練（RTX 3050 4 GB 建議值）：

    docker compose run --rm hw5 python hw05_teacher_baseline.py \
        --max-epoch 30 --max-tokens 2048 --accum-steps 8

這條實驗使用獨立目錄 ``checkpoints/teacher_gru_baseline``，不會覆蓋先前由
``fairseq-train --arch lstm`` 產生的 ``checkpoints/rnn_baseline``。
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import iterators
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
)
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


LOGGER = logging.getLogger("hw5.teacher-baseline")


def parse_args() -> argparse.Namespace:
    """集中管理實驗參數，讓每次實驗可以由指令重現。"""
    parser = argparse.ArgumentParser(
        description="老師式 GRU + Attention Seq2Seq 教學 baseline"
    )
    parser.add_argument("--data", default="DATA/data-bin/ted2020")
    parser.add_argument("--save-dir", default="checkpoints/teacher_gru_baseline")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--max-epoch", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--keep-last-epochs", type=int, default=5)
    parser.add_argument("--resume", default="checkpoint_last.pt")
    parser.add_argument(
        "--no-resume", action="store_true", help="忽略既有 checkpoint，從 epoch 1 開始"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="只跑一個 batch 的 forward/backward，不進行完整訓練",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="強制使用 CPU（只適合除錯）"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定教學實驗的隨機來源，方便比較兩次修改。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_task(args: argparse.Namespace) -> TranslationTask:
    """請 fairseq 讀取已經 preprocess 完成的 .bin/.idx 與字典。"""
    task_cfg = TranslationConfig(
        data=args.data,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        train_subset="train",
        required_seq_len_multiple=8,
        dataset_impl="mmap",
        upsample_primary=1,
    )
    task = TranslationTask.setup_task(task_cfg)
    task.load_dataset("train", epoch=1, combine=True)
    task.load_dataset("valid", epoch=1)
    return task


def get_epoch_iterator(
    task: TranslationTask,
    split: str,
    epoch: int,
    max_tokens: int,
    num_workers: int,
    seed: int,
):
    """依 token 數組 batch；同一 batch 內會自動 padding。"""
    return task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(task.max_positions(), max_tokens),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=True,
    )


class RNNEncoder(FairseqEncoder):
    """老師的雙向 GRU Encoder。

    輸入 ``src_tokens`` 的形狀為 [batch, source_length]。Embedding 後逐步讀取
    英文序列。雙向 GRU 的 forward/backward state 會串接，因此輸出維度是
    ``2 * encoder_hidden_dim``。
    """

    def __init__(self, cfg: Namespace, dictionary, embed_tokens: nn.Embedding):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens
        self.hidden_dim = cfg.encoder_hidden_dim
        self.num_layers = cfg.encoder_layers
        self.padding_idx = dictionary.pad()
        self.dropout_in = nn.Dropout(cfg.dropout)
        self.rnn = nn.GRU(
            input_size=cfg.encoder_embed_dim,
            hidden_size=cfg.encoder_hidden_dim,
            num_layers=cfg.encoder_layers,
            dropout=cfg.dropout if cfg.encoder_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout_out = nn.Dropout(cfg.dropout)

    def _combine_bidirectional_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """[layers*2, B, H] -> [layers, B, H*2]。"""
        layers_times_directions, batch_size, hidden_dim = hidden.shape
        assert layers_times_directions == self.num_layers * 2
        hidden = hidden.view(self.num_layers, 2, batch_size, hidden_dim)
        hidden = hidden.transpose(1, 2).contiguous()
        return hidden.view(self.num_layers, batch_size, hidden_dim * 2)

    def forward(self, src_tokens, src_lengths=None, **unused):
        del src_lengths, unused
        batch_size = src_tokens.size(0)

        # [B, S] -> [B, S, E] -> [S, B, E]
        embedded = self.dropout_in(self.embed_tokens(src_tokens)).transpose(0, 1)
        initial_hidden = embedded.new_zeros(
            self.num_layers * 2, batch_size, self.hidden_dim
        )
        outputs, final_hidden = self.rnn(embedded, initial_hidden)
        outputs = self.dropout_out(outputs)
        final_hidden = self._combine_bidirectional_hidden(final_hidden)

        # True 代表 padding，Attention 計分時必須遮掉。
        padding_mask = src_tokens.eq(self.padding_idx).transpose(0, 1)
        return outputs, final_hidden, padding_mask

    def reorder_encoder_out(self, encoder_out, new_order):
        """Beam search 改排候選順序時，同步改排 Encoder output。"""
        outputs, hidden, padding_mask = encoder_out
        return (
            outputs.index_select(1, new_order),
            hidden.index_select(1, new_order),
            padding_mask.index_select(1, new_order),
        )


class AttentionLayer(nn.Module):
    """Decoder-to-Encoder dot-product Attention。

    每個 Decoder 輸入位置都拿自己的向量去查詢所有 Encoder outputs：
    score -> mask padding -> softmax -> weighted sum -> context。
    """

    def __init__(self, decoder_embed_dim: int, encoder_output_dim: int):
        super().__init__()
        self.query_projection = nn.Linear(
            decoder_embed_dim, encoder_output_dim, bias=False
        )
        self.output_projection = nn.Linear(
            decoder_embed_dim + encoder_output_dim,
            decoder_embed_dim,
            bias=False,
        )

    def forward(
        self,
        decoder_inputs: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [T,B,E] -> [B,T,E]；[S,B,H] -> [B,S,H]
        decoder_batch = decoder_inputs.transpose(0, 1)
        encoder_batch = encoder_outputs.transpose(0, 1)
        padding_batch = encoder_padding_mask.transpose(0, 1)

        query = self.query_projection(decoder_batch)
        scores = torch.bmm(query, encoder_batch.transpose(1, 2))  # [B,T,S]
        scores = scores.float().masked_fill(
            padding_batch.unsqueeze(1), float("-inf")
        ).type_as(scores)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, encoder_batch)

        combined = torch.cat([context, decoder_batch], dim=-1)
        attended = torch.tanh(self.output_projection(combined))
        return attended.transpose(0, 1), weights


class RNNDecoder(FairseqIncrementalDecoder):
    """老師的單向 GRU Decoder，支援 teacher forcing 與增量解碼。"""

    def __init__(self, cfg: Namespace, dictionary, embed_tokens: nn.Embedding):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens
        self.embed_dim = cfg.decoder_embed_dim
        self.hidden_dim = cfg.decoder_hidden_dim
        self.num_layers = cfg.decoder_layers
        self.dropout_in = nn.Dropout(cfg.dropout)
        self.dropout_out = nn.Dropout(cfg.dropout)

        assert cfg.decoder_layers == cfg.encoder_layers
        assert cfg.decoder_hidden_dim == cfg.encoder_hidden_dim * 2

        self.attention = AttentionLayer(
            decoder_embed_dim=cfg.decoder_embed_dim,
            encoder_output_dim=cfg.decoder_hidden_dim,
        )
        self.rnn = nn.GRU(
            input_size=cfg.decoder_embed_dim,
            hidden_size=cfg.decoder_hidden_dim,
            num_layers=cfg.decoder_layers,
            dropout=cfg.dropout if cfg.decoder_layers > 1 else 0.0,
        )
        self.hidden_to_embed = nn.Linear(
            cfg.decoder_hidden_dim, cfg.decoder_embed_dim
        )
        self.output_projection = nn.Linear(
            cfg.decoder_embed_dim, len(dictionary), bias=False
        )
        if cfg.share_decoder_input_output_embed:
            self.output_projection.weight = self.embed_tokens.weight

    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state=None,
        **unused,
    ):
        del unused
        encoder_outputs, encoder_hidden, encoder_padding_mask = encoder_out

        # 訓練時一次輸入完整的右移 target；生成時只計算最新 token。
        if incremental_state is not None:
            cached = self.get_incremental_state(incremental_state, "cached_state")
            if cached is not None:
                prev_output_tokens = prev_output_tokens[:, -1:]
                previous_hidden = cached["previous_hidden"]
            else:
                previous_hidden = encoder_hidden
        else:
            previous_hidden = encoder_hidden

        embedded = self.dropout_in(self.embed_tokens(prev_output_tokens))
        decoder_inputs = embedded.transpose(0, 1)
        attended, attention_weights = self.attention(
            decoder_inputs, encoder_outputs, encoder_padding_mask
        )
        outputs, final_hidden = self.rnn(attended, previous_hidden)
        outputs = self.dropout_out(outputs)
        outputs = self.hidden_to_embed(outputs)
        logits = self.output_projection(outputs).transpose(0, 1)

        if incremental_state is not None:
            self.set_incremental_state(
                incremental_state,
                "cached_state",
                {"previous_hidden": final_hidden},
            )
        return logits, {"attn": attention_weights}

    def reorder_incremental_state(self, incremental_state, new_order):
        cached = self.get_incremental_state(incremental_state, "cached_state")
        if cached is None:
            return
        cached["previous_hidden"] = cached["previous_hidden"].index_select(
            1, new_order
        )
        self.set_incremental_state(incremental_state, "cached_state", cached)


class Seq2Seq(FairseqEncoderDecoderModel):
    """只負責把 Encoder output 接給 Decoder。"""

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **unused):
        del unused
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        return self.decoder(prev_output_tokens, encoder_out=encoder_out)


def initialize_parameters(module: nn.Module) -> None:
    """沿用老師範例的初始化策略。"""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])
    elif isinstance(module, nn.RNNBase):
        for parameter in module.parameters():
            nn.init.uniform_(parameter, -0.1, 0.1)


def build_model(task: TranslationTask) -> Seq2Seq:
    """在本檔案中親自建立老師的 Encoder、Attention 與 Decoder。"""
    cfg = Namespace(
        encoder_embed_dim=256,
        encoder_hidden_dim=512,
        encoder_layers=1,
        decoder_embed_dim=256,
        decoder_hidden_dim=1024,  # 雙向 Encoder 的 512 * 2
        decoder_layers=1,
        share_decoder_input_output_embed=True,
        dropout=0.3,
    )
    source_dictionary = task.source_dictionary
    target_dictionary = task.target_dictionary
    encoder_embedding = nn.Embedding(
        len(source_dictionary), cfg.encoder_embed_dim, source_dictionary.pad()
    )
    decoder_embedding = nn.Embedding(
        len(target_dictionary), cfg.decoder_embed_dim, target_dictionary.pad()
    )
    model = Seq2Seq(
        RNNEncoder(cfg, source_dictionary, encoder_embedding),
        RNNDecoder(cfg, target_dictionary, decoder_embedding),
    )
    model.apply(initialize_parameters)
    return model


class LabelSmoothedCrossEntropy(nn.Module):
    """老師範例手寫的 label-smoothed cross entropy。"""

    def __init__(self, smoothing: float, padding_index: int):
        super().__init__()
        self.smoothing = smoothing
        self.padding_index = padding_index

    def forward(self, log_probabilities, target):
        target = target.unsqueeze(-1)
        negative_log_likelihood = -log_probabilities.gather(-1, target)
        smooth_loss = -log_probabilities.sum(-1, keepdim=True)
        padding_mask = target.eq(self.padding_index)
        negative_log_likelihood.masked_fill_(padding_mask, 0.0)
        smooth_loss.masked_fill_(padding_mask, 0.0)
        epsilon_per_class = self.smoothing / log_probabilities.size(-1)
        return (
            (1.0 - self.smoothing) * negative_log_likelihood.sum()
            + epsilon_per_class * smooth_loss.sum()
        )


class NoamOptimizer:
    """AdamW 外加 warmup + inverse-square-root learning-rate scheduler。"""

    def __init__(self, parameters, model_size=256, factor=2.0, warmup=4000):
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=0.0,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.0001,
        )
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.step_number = 0

    def learning_rate(self, step: Optional[int] = None) -> float:
        step = self.step_number if step is None else step
        if step == 0:
            return 0.0
        return self.factor * self.model_size ** -0.5 * min(
            step ** -0.5, step * self.warmup ** -1.5
        )

    def prepare_next_step(self) -> None:
        """先更新 step 與 LR；真正的 AdamW step 交給 GradScaler 執行。"""
        self.step_number += 1
        rate = self.learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = rate

    def state_dict(self) -> Dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_number": self.step_number,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.step_number = state["step_number"]


def move_sample(sample: Dict, device: torch.device) -> Dict:
    return utils.move_to_cuda(sample, device=device) if device.type == "cuda" else sample


def compute_loss(model, criterion, sample):
    logits, _ = model(**sample["net_input"])
    log_probabilities = F.log_softmax(logits, dim=-1)
    loss = criterion(
        log_probabilities.view(-1, log_probabilities.size(-1)),
        sample["target"].view(-1),
    )
    return loss


def train_one_epoch(
    model,
    task,
    criterion,
    optimizer,
    scaler,
    device,
    args,
    epoch: int,
) -> float:
    """明確展開 forward -> loss -> backward -> optimizer step。"""
    epoch_iterator = get_epoch_iterator(
        task, "train", epoch, args.max_tokens, args.num_workers, args.seed
    )
    batches = epoch_iterator.next_epoch_itr(shuffle=True)
    grouped_batches = iterators.GroupedIterator(batches, args.accum_steps)

    model.train()
    losses: List[float] = []
    progress = tqdm(grouped_batches, desc=f"train epoch {epoch}")
    for samples in progress:
        optimizer.optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_tokens = 0

        for sample in samples:
            sample = move_sample(sample, device)
            with autocast(enabled=device.type == "cuda"):
                loss = compute_loss(model, criterion, sample)
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            accumulated_tokens += sample["ntokens"]

        # Loss 是 token sum，所以更新前把 gradient 除以這次累積的 token 數。
        scaler.unscale_(optimizer.optimizer)
        scale = 1.0 / max(accumulated_tokens, 1)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.prepare_next_step()
        scaler.step(optimizer.optimizer)
        scaler.update()

        token_loss = accumulated_loss / max(accumulated_tokens, 1)
        losses.append(token_loss)
        progress.set_postfix(
            loss=f"{token_loss:.4f}",
            lr=f"{optimizer.learning_rate():.2e}",
            grad=f"{float(gradient_norm):.2f}",
        )
    return float(np.mean(losses))


def decode(tokens, dictionary) -> str:
    text = dictionary.string(tokens.int().cpu(), "sentencepiece")
    return text if text else "<unk>"


@torch.no_grad()
def validate(model, task, criterion, generator, device, args) -> Dict:
    """同時計算 validation loss、中文 BLEU，並保留翻譯範例。"""
    epoch_iterator = get_epoch_iterator(
        task, "valid", 1, args.max_tokens, args.num_workers, args.seed
    )
    batches = epoch_iterator.next_epoch_itr(shuffle=False)

    model.eval()
    losses: List[float] = []
    sources: List[str] = []
    hypotheses: List[str] = []
    references: List[str] = []

    for sample in tqdm(batches, desc="validation"):
        sample = move_sample(sample, device)
        loss = compute_loss(model, criterion, sample) / sample["ntokens"]
        losses.append(loss.item())

        generated = generator.generate([model], sample)
        for index, candidates in enumerate(generated):
            sources.append(
                decode(
                    utils.strip_pad(
                        sample["net_input"]["src_tokens"][index],
                        task.source_dictionary.pad(),
                    ),
                    task.source_dictionary,
                )
            )
            hypotheses.append(decode(candidates[0]["tokens"], task.target_dictionary))
            references.append(
                decode(
                    utils.strip_pad(
                        sample["target"][index], task.target_dictionary.pad()
                    ),
                    task.target_dictionary,
                )
            )

    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="zh")
    return {
        "loss": float(np.mean(losses)),
        "bleu": float(bleu.score),
        "sources": sources,
        "hypotheses": hypotheses,
        "references": references,
    }


def save_checkpoint(
    path: Path, model, optimizer, epoch: int, stats: Dict, best_bleu: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stats": {
                "loss": stats["loss"],
                "bleu": stats["bleu"],
                "best_bleu": best_bleu,
            },
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer, device) -> Tuple[int, float]:
    """回傳下一個 epoch 與歷史最佳 BLEU。"""
    if not path.exists():
        LOGGER.info("找不到 checkpoint，從 epoch 1 開始：%s", path)
        return 1, float("-inf")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    stats = checkpoint.get("stats", {})
    LOGGER.info(
        "續接 %s：已完成 epoch %s，valid loss=%s，BLEU=%s",
        path,
        checkpoint["epoch"],
        stats.get("loss"),
        stats.get("bleu"),
    )
    return checkpoint["epoch"] + 1, float(stats.get("best_bleu", stats.get("bleu", -1)))


def run_smoke_test(model, task, criterion, optimizer, scaler, device, args) -> None:
    """用一個真實 train batch 驗證模型連接、Loss 與 backward。"""
    epoch_iterator = get_epoch_iterator(
        task, "train", 1, min(args.max_tokens, 512), 0, args.seed
    )
    sample = next(epoch_iterator.next_epoch_itr(shuffle=False))
    sample = move_sample(sample, device)
    model.train()
    optimizer.optimizer.zero_grad()
    with autocast(enabled=device.type == "cuda"):
        loss = compute_loss(model, criterion, sample) / sample["ntokens"]
    scaler.scale(loss).backward()
    LOGGER.info("Smoke test 成功：batch tokens=%d, loss=%.4f", sample["ntokens"], loss)


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
        raise RuntimeError("找不到 CUDA GPU；若只想除錯，請明確加上 --cpu")

    LOGGER.info("device=%s", device)
    LOGGER.info("data=%s", Path(args.data).resolve())
    task = setup_task(args)
    model = build_model(task).to(device)
    criterion = LabelSmoothedCrossEntropy(
        smoothing=0.1, padding_index=task.target_dictionary.pad()
    ).to(device)
    optimizer = NoamOptimizer(model.parameters())
    scaler = GradScaler(enabled=device.type == "cuda")

    LOGGER.info("model=%s", model)
    LOGGER.info("trainable parameters=%s", f"{sum(p.numel() for p in model.parameters()):,}")

    if args.smoke_test:
        run_smoke_test(model, task, criterion, optimizer, scaler, device, args)
        return

    save_dir = Path(args.save_dir)
    start_epoch, best_bleu = 1, float("-inf")
    if not args.no_resume:
        start_epoch, best_bleu = load_checkpoint(
            save_dir / args.resume, model, optimizer, device
        )

    generation_args = Namespace(
        beam=args.beam,
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
    generator = task.build_generator([model], generation_args)

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
        if stats["hypotheses"]:
            LOGGER.info("source: %s", stats["sources"][0])
            LOGGER.info("hypothesis: %s", stats["hypotheses"][0])
            LOGGER.info("reference: %s", stats["references"][0])

        is_best = stats["bleu"] > best_bleu
        best_bleu = max(best_bleu, stats["bleu"])
        epoch_path = save_dir / f"checkpoint{epoch}.pt"
        save_checkpoint(epoch_path, model, optimizer, epoch, stats, best_bleu)
        shutil.copy2(epoch_path, save_dir / "checkpoint_last.pt")
        if is_best:
            shutil.copy2(epoch_path, save_dir / "checkpoint_best.pt")

        old_path = save_dir / f"checkpoint{epoch - args.keep_last_epochs}.pt"
        if old_path.exists():
            old_path.unlink()


if __name__ == "__main__":
    main()
