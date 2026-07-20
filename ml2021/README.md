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
