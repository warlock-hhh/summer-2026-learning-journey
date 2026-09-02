# ML2021 Notes

本目錄不只保存分數，而是保存從資料、模型、Loss、Optimization 到 Evaluation 的完整學習過程。大型資料集、模型權重與 Kaggle submission 不納入 Git；依各專案 README 取得資料後即可重跑。

| Project | Task | Best result | Reproduction |
|---|---|---:|---|
| [HW1](hw01-covid-regression/README.md) | Tabular regression | Private RMSE 0.92892 | Ridge baseline |
| [HW2](hw02-phoneme-classification/README.md) | Phoneme classification | Private Acc. 0.73610 | PyTorch MLP |
| [HW3](hw03-food11-classification/README.md) | Image classification | Private Acc. 76.748% | CNN + dynamic pseudo-label |
| [HW4](hw04-speaker-classification/README.md) | Speaker classification | Private Acc. 0.91000 | Transformer encoder |
| [HW5](hw05-machine-translation/README.md) | English-to-Chinese translation | Local BLEU 23.59 | Docker + Transformer |

整體觀念整理：[ML 大架構複習筆記](ML大架構複習筆記.md)


## HW1：COVID-19 Cases Prediction

- 類型：Tabular regression
- 指標：RMSE，越低越好
- 最佳模型：Ridge Regression
- Public／Private：`0.89396`／`0.92892`
- 結論：小型表格資料不一定適合 DNN；baseline、特徵尺度與 validation 設計比盲目增加模型複雜度重要。

## HW2：TIMIT Framewise Phoneme Classification

- 類型：39 類 framewise classification
- 輸入：11 frames × 39 features = 429 dimensions
- Loss：CrossEntropyLoss
- 最佳 Public／Private accuracy：`0.73628`／`0.73610`

### 實驗演進

| 階段 | 主要改動 | Public accuracy |
|---|---|---:|
| Pipeline check | 1 epoch | 0.55417 |
| Sigmoid baseline | 20 epochs | 0.68917 |
| Activation | ReLU | 0.69698 |
| Regularization | Weight decay | 0.70119 |
| Regularization | Dropout 0.2 | 0.71453 |
| Longer training | Dropout 0.2, 15 epochs | 0.72113 |
| Normalization | BatchNorm | 0.72254 |
| Optimization | LR scheduler | 0.72870 |
| Capacity | Wider MLP | 0.73074 |
| Best result | Dropout 0.3 | **0.73628** |

模型權重、資料集、虛擬環境與 submission CSV 不納入 repository；只保存可閱讀的實驗設定、程式與結果摘要。

## HW3：Food-11 Image Classification

- 類型：11 類圖片分類
- 資料：3,080 張 labeled training images、660 張 validation images、3,347 張 testing images
- 原始 baseline Public accuracy：`48.387%`
- 最佳 validation accuracy：`76.36%`
- 最佳 Public／Private accuracy：`77.897%`／`76.748%`
- Public accuracy 相較 baseline 提升約 `29.51` 個百分點

### 實驗演進

| 階段 | 主要方法 | Public accuracy |
|---|---|---:|
| Baseline | 原始 CNN | 48.387% |
| Regularization | Augmentation、Dropout 0.1、scheduler | 55.794% |
| Semi-supervised | GAP + pseudo-label | 64.277% |
| Architecture | Residual + pseudo-label | 64.695% |
| Ensemble | GAP + Residual + CutMix | 68.100% |
| Final | 五層 CNN + dynamic self-labeling | **77.897%** |

### 最終方法

- 五層 CNN，搭配 BatchNorm、ReLU、MaxPool 與 fully connected Dropout。
- 使用 RandomResizedCrop、AutoAugment、HorizontalFlip、ColorJitter 與 RandomAffine。
- validation accuracy 超過 70% 後啟動 dynamic pseudo-label。
- 只採用 confidence ≥ `0.90` 的 unlabeled predictions，每 5 epochs 更新一次。
- 共訓練 500 epochs，最佳 checkpoint 位於 epoch 443，約耗時 8 小時。

核心結論：強 augmentation 能降低小資料集的過擬合，但 pseudo-label 必須先有可靠 teacher，並以 confidence threshold 控制錯誤標籤。Ensemble、TTA 與 CutMix 不保證有效，仍需用公平的 validation 實驗判斷。

## HW4：Speaker Classification with Self-Attention

- 任務：使用 `(T, 40)` log-Mel 特徵進行 600 類語者分類。
- 資料：69,438 筆訓練特徵、6,000 筆測試特徵；固定 `seed=87` 做 90%／10% train-validation split。
- 課程 baseline Public／Private：`0.76428`／`0.75888`。
- 最佳 validation accuracy／loss：`86.85%`／`0.5565`。
- 最終 Public／Private：`0.91833`／`0.91000`。

### 實驗演進

| 階段 | 設定 | Valid Acc | Public | Private |
|---|---|---:|---:|---:|
| E1 | 1 layer、2 heads、segment 128 | 60.46% | 0.73690 | 0.72166 |
| E2 | Encoder layer 1 → 2 | 66.13% | 0.79904 | 0.80000 |
| E3 | Attention heads 2 → 4 | 69.20% | 0.83285 | 0.81000 |
| E4 | Segment 128 → 192 | 76.63% | 0.86023 | 0.84666 |
| E5 | Segment 192 → 256 | 81.41% | 0.86714 | 0.85777 |
| E6 | Epochs 20 → 30 | 83.73% | 0.89238 | 0.89611 |
| E7 | Mean → Attention Pooling | **86.85%** | **0.91833** | **0.91000** |

### 最終架構

```text
Input (T, 40)
→ Linear Projection 40 → 80
→ 2 Transformer Encoder Layers, 4 Heads
→ Attention Pooling 80 → 64 → 1
→ Classifier 80 → 80 → 600
```

訓練使用 segment length 256、batch size 16、30 epochs、AdamW、warmup 與 cosine decay。實驗顯示更長語音上下文能改善聲紋辨識，但 Self-Attention 成本約隨序列長度平方成長；Attention Pooling 則能避免重要 frame 被等權平均稀釋。

## HW5：English-to-Traditional-Chinese Translation

- 任務：TED2020 英文翻譯成繁體中文。
- 訓練／驗證資料：390,041／3,939 組平行句對。
- Tokenization：SentencePiece，8,000 joined vocabulary。
- 執行環境：Windows、WSL2、Docker、RTX 3050 Laptop GPU 4 GB。
- 評估限制：JudgeBoi 已失效，只能報告 local validation BLEU。

### 模型比較

| 模型 | 最佳 Epoch | Local Validation BLEU |
|---|---:|---:|
| 老師式 GRU + Attention | 28 | 18.60 |
| fairseq LSTM | 40 | 20.64 |
| 4-layer Transformer | **38** | **23.59** |

最終 Transformer 使用 4-layer Encoder／Decoder、`d_model=256`、4 heads、FFN 1024、Pre-LayerNorm、label smoothing、warmup、inverse-square-root scheduler、FP16、gradient accumulation 與 beam search 5。

核心結論：GRU、LSTM 與 Transformer 都屬於 Seq2Seq Encoder–Decoder；差別在序列資訊的傳遞方式。Transformer 以 self-attention 建立全句關係，將 BLEU 從 GRU 的 18.60 提升到 23.59，但 BLEU 仍須搭配人工案例檢查語意錯譯。
