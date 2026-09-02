# ML2021 HW5：English-to-Traditional-Chinese Seq2Seq

這個 repository 記錄 ML2021 HW5 英翻中實驗，重點是從 recurrent Seq2Seq baseline，逐步理解並實作 Transformer Medium Baseline，同時建立可在 Windows、Docker Desktop 與 RTX 3050 上重現的訓練環境。

## 實驗結果

所有分數皆為相同本機 validation split、SacreBLEU `tokenize='zh'` 的結果，不是當年 JudgeBoi hidden test 成績。

| 模型 | 最佳 Epoch | Local validation BLEU |
|---|---:|---:|
| 老師式 GRU + Attention | 28 | 18.60 |
| fairseq 內建 LSTM | 40 | 20.64 |
| **4-layer Transformer** | **38** | **23.59** |

Transformer 相對 GRU 提升 4.99 BLEU，接近課程 Medium Baseline 的分數區間。

## 最終 Transformer

```text
Encoder / Decoder : 4 layers / 4 layers
d_model           : 256
Attention heads   : 4
FFN dimension     : 1,024
Dropout           : 0.3
Label smoothing   : 0.1
Optimizer         : AdamW
Scheduler         : 4,000-step warmup + inverse-square-root decay
Training          : FP16, gradient accumulation, beam=5
```

## 專案結構

| 檔案 | 用途 |
|---|---|
| `HW5_研究日誌.md` | 完整實驗流程、核心技術、結果與重現方式 |
| `preprocess_hw5.py` | 資料清理、切分與 SentencePiece 前處理 |
| `hw05_teacher_baseline.py` | 可閱讀的 GRU Encoder–Attention–Decoder 教學版 |
| `hw05_transformer_medium.py` | 4-layer Transformer Medium Baseline |
| `train_teacher_baseline.ps1` | 從 Windows/Docker 啟動 GRU 訓練 |
| `train_transformer_medium.ps1` | 從 Windows/Docker 啟動 Transformer 訓練 |
| `train_baseline.ps1` | fairseq 內建 LSTM baseline 啟動器 |
| `Dockerfile`、`docker-compose.yml` | 固定 PyTorch、CUDA runtime 與 fairseq 環境 |

DATA、checkpoint、validation outputs、虛擬環境、課程 PDF 與老師原始範例不納入 repository。

## 執行環境

- Windows + WSL2 + Docker Desktop
- NVIDIA RTX 3050 Laptop GPU 4 GB
- Python 3.7（Container）
- PyTorch 1.10.0 + CUDA runtime 11.3
- fairseq commit `9a1c497`

## 快速驗證

先啟動 Docker Desktop，再於專案根目錄執行：

```powershell
.\train_transformer_medium.ps1 -SmokeTest
```

Smoke test 只跑一個真實 batch，用來確認資料、CUDA、Forward、Loss、Backward 與 GPU 記憶體。

## 正式訓練

```powershell
.\train_transformer_medium.ps1 -MaxEpoch 40
```

若已有 `checkpoint_last.pt`，預設會接續訓練；從頭開始可加上 `-NoResume`。

RTX 3050 發生 OOM 時：

```powershell
.\train_transformer_medium.ps1 -MaxEpoch 40 -MaxTokens 1024 -AccumSteps 12
```

資料取得與完整前處理背景請參考研究日誌；本 repository 不重新散布課程資料。

## 評估限制

原作業的 JudgeBoi 已無法使用，公開 test 中文內容是佔位符，因此不能計算官方 test BLEU。本專案只報告 held-out validation BLEU，並以實際翻譯案例輔助分析。

## 來源與致謝

本專案基於李宏毅老師 ML2021 Spring HW05 Sequence-to-Sequence 作業進行學習與重整。模型教學結構與作業提示參考原課程材料：

- [ML2021 Spring HW05](https://github.com/ga642381/ML2021-Spring/tree/main/HW05)
- [fairseq](https://github.com/facebookresearch/fairseq)

本 repository 的 Docker 化、Windows 啟動腳本、可執行教學版、Transformer 實驗流程與研究日誌為本次本機實驗整理成果。
