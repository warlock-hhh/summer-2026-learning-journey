# ML2021 Spring HW2 研究日誌

- 日期：2026-07-19
- 題目：TIMIT Framewise Phoneme Classification
- 任務類型：39 類多類別分類
- 目前最佳 Kaggle 成績：Public 0.73628／Private 0.73610
- 最佳程式版本：`b4caca7 experiment: increase dropout to 0.3, public 0.73628`

## 一、研究目標

本作業使用 TIMIT 語音資料，目標是根據某一時間點附近的聲學特徵，判斷中央音框屬於哪一個 phoneme（音素）。每筆輸入有 429 個特徵：

```text
429 = 11 個相鄰音框 × 每個音框 39 個聲學特徵
```

模型輸出 39 個 logits，分別對應 39 種音素。利用 `CrossEntropyLoss` 比較 logits 與正確類別，最後以 `argmax` 選出預測音素。

## 二、資料與實驗環境

### 資料規模

| 資料 | Shape | 說明 |
|---|---:|---|
| 完整訓練資料 | `(1,229,932, 429)` | 有標籤的音框資料 |
| 實際 training set | `(983,945, 429)` | 前 80% |
| validation set | `(245,987, 429)` | 後 20% |
| testing set | `(451,552, 429)` | Kaggle 預測資料 |

### 執行環境

- 作業系統：Windows
- GPU：NVIDIA RTX 3050 Laptop GPU（4 GB）
- PyTorch：2.6.0+cu124
- Python：3.12.5 虛擬環境
- 訓練裝置：CUDA
- Batch size：64
- Random seed：0

## 三、Baseline 模型

助教範例使用三層 MLP：

```text
429 → 1024 → 512 → 128 → 39
```

最初使用 Sigmoid，optimizer 為 Adam，loss 為 CrossEntropyLoss。

初次執行先以 1 epoch 確認資料、模型、訓練與 submission pipeline 能完整跑通，Public 約為 0.55417。這一步的重點不是追求成績，而是確認 baseline 可執行。

## 四、實驗設計原則

每次只改一個主要變因，其他條件盡量固定，並利用 Git 保存有效版本。主要判斷依據包括：

1. Training accuracy／loss
2. Validation accuracy／loss
3. Kaggle Public score
4. Kaggle Private score
5. 是否出現 overfitting

## 五、實驗結果

| 階段 | 主要修改 | Epoch | Private | Public | 結果判讀 |
|---:|---|---:|---:|---:|---|
| 0 | 初次 pipeline 試跑 | 1 | 0.55426 | 0.55417 | 確認程式可完整執行 |
| 1 | Sigmoid baseline | 5 | 0.67219 | 0.67284 | 增加訓練輪數後大幅改善 |
| 2 | Sigmoid baseline | 20 | 0.68869 | 0.68917 | 後期效益有限，開始過擬合 |
| 3 | Sigmoid 改為 ReLU | 5 | 0.69713 | 0.69698 | ReLU 收斂更快，明顯優於 Sigmoid |
| 4 | ReLU | 20 | 0.69987 | 0.69880 | 比 5 epoch 僅小幅提升，後期過擬合 |
| 5 | 加入 weight decay `1e-4` | 10 | 0.70065 | 0.70119 | 正規化帶來小幅且一致的改善 |
| 6 | 加入 Dropout `0.2` | 10 | 0.71367 | 0.71453 | 有效降低過擬合，提升幅度明顯 |
| 7 | Dropout `0.2` 延長訓練 | 15 | 0.71915 | 0.72113 | 正規化後可安全訓練更久 |
| 8 | 加入 BatchNorm | 15 | 0.72132 | 0.72254 | Val loss 與 Kaggle 成績均小幅改善 |
| 9 | 加入 LR scheduler | 25 | 0.72748 | 0.72870 | 停滯後降低 LR，後期再次改善 |
| 10 | 加寬 MLP | 25 | 0.72947 | 0.73074 | 容量增加有效，但第 16 epoch 後明顯過擬合 |
| 11 | Dropout `0.2 → 0.3` | 25 | **0.73610** | **0.73628** | 抑制加寬模型過擬合，得到目前最佳結果 |

從初次 Public 0.55417 到目前 0.73628，共提升：

```text
0.73628 - 0.55417 = 0.18211
```

也就是約 **18.21 個百分點**。

## 六、各項方法的技術分析

### 1. Sigmoid 改為 ReLU

Sigmoid 在輸入絕對值較大時梯度會接近 0，深層網路容易出現 gradient vanishing。ReLU 在正區間梯度固定，較容易傳遞梯度，因此只跑 5 epoch 就超過 Sigmoid 20 epoch。

### 2. Weight Decay

Adam 設定 `weight_decay=1e-4`，限制模型權重持續變大，避免模型使用過度複雜的決策邊界記住訓練資料。單獨提升不大，但成為後續正規化組合的一部分。

### 3. Dropout

訓練時隨機關閉部分神經元，避免模型過度依賴特定神經元組合。驗證與推論時 `model.eval()` 會關閉 Dropout，因此可能發生 Train loss 高於 Val loss、Train accuracy 低於 Val accuracy，這不一定是錯誤。

原模型使用 Dropout 0.2 已有明顯效果。模型加寬為原本約 3.4 倍後，Dropout 0.2 不足以控制 overfitting；提高為 0.3 後，Private 與 Public 分別提升 0.00663 與 0.00554。

### 4. Batch Normalization

目前每層順序為：

```text
Linear → BatchNorm → ReLU → Dropout
```

BatchNorm 將 mini-batch 中間特徵整理到較穩定的尺度，再透過可學習的 scale 與 bias 保留表達能力。加入後 Public 從 0.72113 提升至 0.72254，幅度不大但三種評估訊號一致改善。

### 5. Learning-rate Scheduler

使用 `ReduceLROnPlateau`：

```text
mode = min
factor = 0.5
patience = 2
```

當 validation loss 停滯後，learning rate 會減半。實驗中 LR 從 `1e-4` 降到 `5e-5` 後，Val loss 再次下降，Public 從 0.72254 提升到 0.72870。

### 6. 加寬 MLP

模型由：

```text
429 → 1024 → 512 → 128 → 39
```

改為：

```text
429 → 2048 → 1024 → 512 → 39
```

參數量由約 104 萬增加到約 353 萬，約為原本 3.4 倍。加寬後能學習更複雜的音素分類規則，最佳 Val accuracy 提升；但訓練後期 Train loss 持續下降、Val loss 上升，證明容量增加也加重 overfitting。

## 七、目前最佳模型

```text
Input: 429
Hidden layers: 2048 → 1024 → 512
Output: 39
Activation: ReLU
BatchNorm: 每個 hidden layer 後
Dropout: 0.3
Loss: CrossEntropyLoss
Optimizer: Adam
Learning rate: 1e-4
Weight decay: 1e-4
Scheduler: ReduceLROnPlateau(factor=0.5, patience=2)
Maximum epochs: 25
Batch size: 64
```

訓練時依 validation accuracy 保存最佳 checkpoint，所以即使後期發生 overfitting，測試階段仍載入表現最佳的模型。

## 八、實作過程遇到的問題

1. 助教 notebook 的 `!gdown`、`!unzip` 是 Colab shell 語法，直接放進 `.py` 會造成 SyntaxError。
2. 相對路徑受到 PowerShell 當前工作目錄影響，後來改用 `Path(__file__)` 建立穩定資料路徑。
3. 新版 NumPy 移除 `np.int`，改為 `np.int64`。
4. 原本 Python 環境只能使用 CPU，另外建立 PyTorch CUDA 虛擬環境後成功使用 GPU。
5. VS Code 曾因外部修改產生 stale buffer，造成存檔衝突；利用 Git 與 Compare/Revert 恢復正確版本。
6. 使用 `.gitignore` 避免提交資料集、checkpoint、prediction CSV、虛擬環境與大型安裝檔，只追蹤主要程式。

## 九、研究結論

1. Activation function 對收斂速度影響明顯，ReLU 遠優於原始 Sigmoid baseline。
2. 單純增加 epoch 很快遇到 diminishing returns，甚至造成 overfitting。
3. Dropout 是本次最有效的正規化手段；模型容量增加後也必須同步加強正規化。
4. BatchNorm 帶來穩定的小幅改善，且使模型較容易訓練。
5. Scheduler 能在固定 LR 停滯後，以較小步伐繼續最佳化。
6. 加寬模型確實提升 capacity，但必須與 Dropout、weight decay、BatchNorm 和最佳 checkpoint 搭配。
7. 最終改善不是來自單一技巧，而是模型容量與正規化之間的平衡。

## 十、Meeting 口頭報告版本

> 這週完成 ML2021 HW2 的 TIMIT 音素分類。每筆輸入由中央音框前後共 11 個 frame 組成，每個 frame 39 維，所以輸入是 429 維，輸出為 39 類 phoneme。我的策略是先跑通助教 baseline，再一次只改一個變因。原本 Sigmoid 收斂較慢，因此先改用 ReLU；接著依序加入 weight decay、Dropout、BatchNorm 與 learning-rate scheduler。模型加寬後 validation accuracy 有提升，但也出現明顯 overfitting，所以把 Dropout 從 0.2 提高到 0.3。最後 Kaggle Public 從初次試跑的 0.55417 提升到 0.73628，Private 為 0.73610。這次最大的觀察是，增加模型容量本身不夠，必須同時用正規化與 scheduler 控制泛化能力。

## 十一、已完成與尚未完成

### 已完成

- 跑通資料載入、訓練、驗證、推論與 Kaggle submission 流程。
- 建立可使用 CUDA 的本機訓練環境。
- 理解 429 維輸入、39 類輸出與 CrossEntropyLoss。
- 完成 ReLU、weight decay、Dropout、BatchNorm、scheduler、模型加寬實驗。
- 使用 Git 保存每個有效實驗版本。
- Kaggle Public 達到 0.73628、Private 達到 0.73610。

### 尚未完成／可延伸

- 將 `11 × 39` 還原成時間序列，嘗試 1D CNN 或 BiLSTM，而不是完全攤平交給 MLP。
- 加入 Early Stopping，避免最佳點後繼續浪費訓練時間；目前最佳 checkpoint 已能防止使用較差的最後一輪模型。
- 測試 Dropout 0.25／0.35，但預期只會是小幅調參改善。
- 使用多個 random seed 檢查提升是否穩定，而不是只依賴 seed 0。
- 檢查目前依資料順序切出的 validation set 是否具有代表性，並避免用 Kaggle Public leaderboard 過度調參。

## 十二、Meeting 可能追問

### Q1：為什麼 429 等於 11 × 39？

因為語音的單一音框資訊不足，所以將中央音框前後的 11 個相鄰音框一起作為輸入；每個音框有 39 個聲學特徵。

### Q2：為什麼 ReLU 比 Sigmoid 好？

Sigmoid 在飽和區域梯度接近零，容易造成梯度消失；ReLU 在正區間能保留較強梯度，因此收斂較快。

### Q3：為什麼 Train loss 可能比 Val loss 高？

因為訓練時 Dropout 開啟，模型刻意在缺少部分神經元的狀態下學習；驗證時 Dropout 關閉，所以驗證條件反而較容易。

### Q4：Scheduler 和 Adam 有什麼差別？

Adam 依每個參數的梯度歷史調整更新尺度；scheduler 調整 optimizer 的整體基礎 learning rate。兩者可以同時使用。

### Q5：加寬模型為什麼同時提升準確率又造成過擬合？

更多神經元提高模型表達能力，也提高記住訓練資料細節的能力。因此需要更強 Dropout 等正規化方法取得平衡。

### Q6：Hessian 作業和 Kaggle 有什麼關係？

HW2-2 使用 Hessian 的特徵值與 gradient norm 判斷模型位於 local-minimum-like、saddle point 或其他位置，是獨立觀念題；不會直接改善 HW2-1 的 Kaggle submission。
