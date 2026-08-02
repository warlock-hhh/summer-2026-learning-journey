# ML2021 Notes

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
