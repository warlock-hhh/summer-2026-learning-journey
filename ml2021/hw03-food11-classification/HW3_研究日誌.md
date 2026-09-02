# ML2021 HW3 研究日誌

## 作業目標

使用 CNN 將 Food-11 圖片分成 11 個類別，並以 validation 與 Kaggle
成績判斷模型的泛化能力。

資料量：

- Training labeled：3,080 張
- Validation：660 張
- Testing：3,347 張

## 實驗結果

| 實驗 | Epoch | 最佳 Valid Acc | Kaggle Private | Kaggle Public |
|---|---:|---:|---:|---:|
| 原始 baseline | 20 | 47.88% | 45.965% | 48.387% |
| Data augmentation | 20 | 51.52% | 49.312% | 48.924% |
| Augmentation + Dropout 0.3 | 20 | 45.61% | 43.335% | 43.966% |
| Augmentation + Dropout 0.3 | 40 | 53.03% | 47.818% | 47.311% |
| Augmentation + Dropout 0.1 | 40 | 55.76% | 52.779% | 51.911% |
| Augmentation + Dropout 0.1 + Scheduler | 40 | **56.21%** | **56.066%** | **55.794%** |
| GAP + Pseudo label | 20 | 64.09% | 64.196% | 64.277% |
| Residual + Pseudo label | 20 | 66.06% | 65.391% | 64.695% |
| GAP + Residual + CutMix Ensemble | - | 68.94% | 68.798% | 68.100% |
| 五層 CNN + Dynamic pseudo-label | 500 | **76.36%** | **76.748%** | **77.897%** |

## 早期最佳設定

```text
Data augmentation：開啟
Dropout：0.1
Optimizer：Adam
Initial learning rate：3e-4
Weight decay：1e-5
Epoch：40
Scheduler：ReduceLROnPlateau
Scheduler factor：0.5
Scheduler patience：3
Minimum learning rate：1e-6
最佳 checkpoint：Epoch 39
```

Scheduler 實際降低 learning rate 的過程：

```text
3.00e-4
1.50e-4
7.50e-5
3.75e-5
1.87e-5
```

相較於相同設定但沒有 Scheduler：

- Private score：52.779% → 56.066%，提升 3.287 個百分點。
- Public score：51.911% → 55.794%，提升 3.883 個百分點。

## 目前觀察

1. 原始 CNN 在少量資料上很快 overfit。
2. Data augmentation 能改善泛化能力。
3. Dropout 0.3 對此模型太強，造成學習速度過慢；0.1 較平衡。
4. 固定 learning rate 使後期 validation 表現波動。
5. 動態降低 learning rate 後，Kaggle Private 與 Public 都明顯提升。
6. 最後一輪 train accuracy 為 79.12%，valid accuracy 為 56.06%，
   仍存在約 23 個百分點的差距，因此 overfitting 尚未完全解決。

## 程式輸出

- 每次實驗保存 validation accuracy 最佳的 checkpoint。
- 訓練完成後以最佳 checkpoint 產生 Kaggle CSV。
- 每個 epoch 更新 `training_curves.png`。
- 曲線圖使用固定檔名覆蓋，避免累積大量圖片。
- 曲線圖同時顯示 train/valid loss、accuracy 與最佳 epoch。

## 中期探索與反省

早期改善集中在 Dropout、scheduler 與 weight decay，但提升逐漸有限。
後續加入 GAP、Residual connection、CutMix、pseudo-label、Ensemble 與
TTA，最高 Public accuracy 約為 68.7%。這些實驗有助於理解不同技巧，
但也一度變成為了分數疊加方法，難以判斷真正的主要改善來源。

因此最後重新參考高分 notebook，將競賽版本獨立成
`hw03_reference.py`，不再繼續堆疊 Ensemble，而是完整重現一套一致的
訓練流程。

## 最終方法：Dynamic self-labeling

### 模型

- 五層 CNN，channel 依序為 64、128、256、512、1024。
- 每層使用 Conv2d、BatchNorm、ReLU 與 MaxPool。
- Fully connected layers 使用 Dropout 0.6 與 0.4。
- 約 43 MB 的最佳模型權重。

### 資料增強

- RandomResizedCrop 128×128。
- AutoAugment（default、CIFAR10、SVHN policy 隨機選擇）。
- HorizontalFlip、ColorJitter、RandomAffine。
- 增強在讀取 batch 時即時產生，不修改硬碟上的原始圖片。

### Optimization

```text
Optimizer：Adam
Learning rate：3e-4
Weight decay：1e-5
Loss：CrossEntropyLoss
Gradient clipping：max norm 10
Batch size：32
Epoch：500
```

### Dynamic pseudo-label 流程

1. 先只使用 3,080 張 labeled images 訓練。
2. Best validation accuracy 超過 70% 後，啟動 self-labeling。
3. Teacher 對 6,786 張 unlabeled images 產生 Softmax confidence。
4. 只選擇 confidence ≥ 0.90 的預測作為 pseudo labels。
5. 將 pseudo-labeled images 暫時與 labeled dataset 合併訓練。
6. 每 5 epochs 重新推論與更新 pseudo labels。

Pseudo-label 在 epoch 190 首次啟動，當時加入 1,783 張圖片；訓練結束
前已選入約 4,287 張。這些標籤不會寫回原始 dataset，而是訓練期間在
記憶體中動態建立。

## 最終結果

```text
完整訓練：500 epochs
最佳 epoch：443
最佳 validation accuracy：76.36%
最佳 validation loss：0.8066
Kaggle Private accuracy：76.748%
Kaggle Public accuracy：77.897%
```

Validation、Private 與 Public accuracy 接近，表示 validation split 能
合理反映模型泛化能力。Public 成績相較最初 baseline 的 48.387% 提升
約 29.51 個百分點，也超過原先設定的 70% 目標。

## 結論與學習重點

1. 高準確率不是由單一超參數造成，而是模型容量、資料增強、資料量與
   optimization 共同作用。
2. 少量 labeled data 很容易 overfit；強 augmentation 能提供不同輸入
   變化，但不會憑空產生新的語意資訊。
3. Pseudo-label 的關鍵不是把所有未標記資料加入，而是先建立可靠的
   teacher，再以 confidence threshold 控制標籤雜訊。
4. Dynamic self-labeling 會隨 teacher 進步更新標籤，比只產生一次
   pseudo labels 更完整。
5. Train accuracy 可能低於 validation accuracy，因為 training data
   使用強增強、Dropout，且包含可能有雜訊的 pseudo labels。
6. Ensemble、TTA 與 CutMix 並非必然提升；方法是否有效仍應由公平的
   validation 實驗驗證。
7. 這次 500 epochs 約訓練 8 小時，主要成本來自大型 CNN、即時強增強、
   pseudo-label 後增加的資料量，以及每 5 epochs 掃描全部 unlabeled data。

本作業到此結案；後續複習以能解釋 model、augmentation、training loop
與 dynamic pseudo-label 四個核心為主，不再繼續以堆疊技巧刷分。
