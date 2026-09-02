# ML2021 HW3 實驗技術完整解析

這份文件回答兩個問題：

1. 從最初 baseline 到最後 Reference 版，每一次實驗究竟改了什麼？
2. 為什麼最後的 Reference 版能明顯超越前面的模型？

重點不是只記住最高分，而是理解每項技術想解決的問題、實際結果與
實驗限制。

---

## 1. 作業問題與資料

HW3 是 Food-11 十一類食物圖片分類。模型輸入一張 RGB 圖片，輸出
11 個 logits，取最大值作為預測類別。

| 資料 | 數量 | 是否有標籤 | 用途 |
|---|---:|---|---|
| Labeled training | 3,080 | 有 | 更新模型參數 |
| Validation | 660 | 有 | 選模型、觀察泛化，不參與參數更新 |
| Unlabeled training | 6,786 | 無 | 後期產生 pseudo-label |
| Testing | 3,347 | 無 | 產生 Kaggle submission |

這份作業最大的困難是 labeled data 只有 3,080 張。對 CNN 而言資料偏
少，因此模型很容易記住 training images，卻無法辨認沒看過的圖片。

### 評估指標

- **Train loss / accuracy**：模型對訓練資料的擬合程度。
- **Validation loss / accuracy**：模型對未參與更新資料的泛化程度。
- **Kaggle Public / Private accuracy**：測試集不同切分的正確率。

Validation 的用途不是讓模型學習，而是回答：「目前學到的規則，對沒
看過的資料是否仍有效？」

---

## 2. 完整實驗成績總覽

下表依實驗演進整理。部分早期 submission 沒有保存對應的最佳 validation
數值，因此只列實際留下的 Kaggle 成績，不補猜測值。

| # | 實驗 | Epoch | Valid Acc | Private | Public |
|---:|---|---:|---:|---:|---:|
| 1 | 原始 baseline | 20 | 47.88% | 45.965% | 48.387% |
| 2 | Data augmentation | 20 | 51.52% | 49.312% | 48.924% |
| 3 | Augmentation + Dropout 0.3 | 20 | 45.61% | 43.335% | 43.966% |
| 4 | Augmentation + Dropout 0.3 | 40 | 53.03% | 47.818% | 47.311% |
| 5 | Augmentation + Dropout 0.1 | 40 | 55.76% | 52.779% | 51.911% |
| 6 | Augmentation + Dropout 0.1 + Scheduler | 40 | 56.21% | 56.066% | 55.794% |
| 7 | 上述設定 + Weight decay 1e-4 | 40 | 57.58% 左右 | 54.273% | 54.778% |
| 8 | 上述設定 + Weight decay 3e-5 | 40 | 57.12% 左右 | 54.991% | 56.152% |
| 9 | 單次 pseudo-label，threshold 0.95 | 20 fine-tune | 57.27% 左右 | 56.664% | 57.347% |
| 10 | 單次 pseudo-label，threshold 0.90 | 20 fine-tune | 57.88% 左右 | 57.441% | 55.854% |
| 11 | GAP architecture | 40 | 62.27% | 61.028% | 60.573% |
| 12 | GAP + pseudo-label 0.90 | 20 fine-tune | 64.09% | 64.196% | 64.277% |
| 13 | Residual architecture | 40 | 65.61% | 62.582% | 62.843% |
| 14 | Residual + pseudo-label 0.95 | 20 fine-tune | 66.06% | 65.391% | 64.695% |
| 15 | Residual + CutMix | 40 | 64.39% 左右 | 61.267% | 62.246% |
| 16 | Residual + CutMix | 80 | 67% 左右 | 67.483% | 65.710% |
| 17 | GAP + Residual Ensemble | - | 約 67% | 67.364% | 67.204% |
| 18 | GAP + Residual + CutMix Ensemble | - | 68.94% | 68.798% | 68.100% |
| 19 | 三模型 Ensemble + horizontal-flip TTA | - | 68.94% | 68.499% | 68.697% |
| 20 | 五層 CNN + Dynamic self-labeling | 500 | **76.36%** | **76.748%** | **77.897%** |

注意：不同實驗未固定 random seed，而且部分實驗同時改變架構、epoch、
augmentation 或 pseudo-label。表格能呈現實際演進，但不是嚴格的消融
實驗；除非一次只改一個變因，否則不能把全部提升歸因於單一技術。

---

## 3. 實驗 1：原始 Baseline

### 做了什麼

最初模型使用三個 convolution blocks：

```text
Input 3×128×128
  → Conv 64 + BatchNorm + ReLU + MaxPool
  → Conv 128 + BatchNorm + ReLU + MaxPool
  → Conv 256 + BatchNorm + ReLU + MaxPool
  → Flatten
  → Fully Connected
  → 11 logits
```

Loss 使用 `CrossEntropyLoss`，optimizer 使用 Adam。模型輸出不先做
Softmax，因為 `CrossEntropyLoss` 內部已包含等價的 log-softmax 計算。

### 想回答的問題

先建立一個能跑完 training、validation、testing 與產生 Kaggle CSV 的
最小系統。Baseline 的作用是提供比較基準，不是追求最高分。

### 結果與解讀

- Validation：47.88%
- Kaggle Public：48.387%

Train accuracy 很快提高，但 validation 明顯落後，表示模型開始記住
3,080 張 training images。核心問題是資料太少、變化不足，而不是單純
epoch 不夠。

---

## 4. 實驗 2：On-the-fly Data Augmentation

### 改用了什麼

每次 DataLoader 讀取圖片時，隨機執行：

- `RandomResizedCrop(scale=0.85~1.0)`
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(10°)`
- 小幅 `ColorJitter`

原圖不會被複製或修改。相同圖片在不同 epoch 可能呈現不同裁切、角度
與光線，因此模型不能只記住固定像素位置。

### 想解決什麼

讓有限的 labeled images 產生更多合理變化，降低 overfitting，並讓模型
學到對位置、角度與亮度較不敏感的特徵。

### 結果與解讀

Validation 從 47.88% 提升到 51.52%，證明資料變化確實有幫助。但
Public 只從 48.387% 變成 48.924%，表示 20 epochs 仍不足以讓模型充分
吸收變難後的 training data。

資料增強不是免費增加正確的新語意，只是對既有圖片做合理變換。若裁切
過度或顏色扭曲過大，也可能破壞食物的重要特徵。

---

## 5. 實驗 3～5：Dropout 強度與訓練時間

### Dropout 的作用

Dropout 在訓練時隨機把部分 hidden activations 設為 0，迫使模型不能
依賴固定少數神經元。它是一種 regularization：通常能降低 overfitting，
但也會讓 optimization 變難、收斂變慢。

### Dropout 0.3，20 epochs

- Public 降到 43.966%。
- 對當時的小模型與短訓練而言，0.3 太強。
- 這比較像 underfitting：模型連 training pattern 都尚未學充分。

### Dropout 0.3，40 epochs

增加訓練時間後 valid accuracy 回升到 53.03%，說明先前不完全是方法
錯誤，也有「regularization 後需要更久才能收斂」的因素。但 Kaggle
Public 仍只有 47.311%，整體不理想。

### Dropout 0.1，40 epochs

降低到 0.1 後，Private 52.779%、Public 51.911%。這代表當時模型的較
合理平衡是：保留少量 regularization，但不要強到阻礙學習。

### 學到的事

Dropout 數值不能脫離模型大小與訓練長度單獨判斷。同樣是 0.3，在小型
模型可能太強，在後來參數更多的 Reference 模型卻可能仍然不足。

---

## 6. 實驗 6：動態 Learning Rate Scheduler

### 改用了什麼

使用 `ReduceLROnPlateau` 觀察 validation loss。如果連續數個 epoch 沒有
改善，就將 learning rate 乘以 0.5：

```text
3.00e-4 → 1.50e-4 → 7.50e-5 → 3.75e-5 → 1.87e-5
```

### 原理

Learning rate 決定 optimizer 每次更新參數的步伐：

- 太大：能快速前進，但接近低點時容易跨過最佳位置而震盪。
- 太小：穩定但非常慢，也可能在有限 epoch 內學不完。

Scheduler 前期用較大步伐學習，後期在 validation 停滯時縮小步伐，做
更細緻的調整。

### 結果

- Private：52.779% → 56.066%
- Public：51.911% → 55.794%

這是相對清楚的改善，因為主要差異是加入 scheduler。但 scheduler 只
改善「怎麼走」，不能解決模型表達能力不足或 labeled data 太少。

---

## 7. 實驗 7～8：Weight Decay

### 改用了什麼

Adam 的 `weight_decay` 從原本 `1e-5` 測試到：

- `1e-4`
- `3e-5`

Weight decay 會在更新時傾向讓權重不要無限制變大，可視為對複雜模型的
懲罰。值越大，regularization 通常越強。

### 結果與解讀

`1e-4` 的 Public 為 54.778%，低於 scheduler baseline；`3e-5` 的
Public 為 56.152%，只比 55.794% 小幅變化。結果表示：

- 過強的 weight decay 可能限制模型擬合能力。
- 這個階段的瓶頸並不主要是 weight 大小。
- 在沒有固定 random seed 的情況下，小於一個百分點的差異不能過度解讀。

---

## 8. 實驗 9～10：第一次使用 Pseudo-label

### Pseudo-label 是什麼

Unlabeled images 沒有人工答案，但已訓練的模型可以先預測：

```text
logits → Softmax → 11 類機率 → 取最大機率類別
```

若最大機率高於 threshold，就把預測類別暫時視為標籤，再與 labeled
data 合併 fine-tune。這屬於 semi-supervised learning。

### 為什麼先做 confidence calibration

Validation 有真實標籤，因此可以檢查「模型說自己有 90% 把握時，實際
到底多常答對」。這比直接相信 Softmax 數字更可靠。

### Threshold 0.95

- 選擇數量較少，但標籤相對可靠。
- Public 提升到 57.347%。

### Threshold 0.90

- 納入更多圖片，但錯誤標籤風險也較高。
- Private 57.441%，Public 55.854%。

兩者沒有一致勝負，反映 threshold 的 precision/coverage trade-off：

- 門檻高：標籤較準、資料較少。
- 門檻低：資料較多、雜訊較大。

這一版只產生一次 pseudo-label，再固定 fine-tune；teacher 之後即使進步，
舊標籤也不會重新更新，是它的重要限制。

---

## 9. 實驗 11～12：Global Average Pooling（GAP）

### 改用了什麼

舊 baseline 把整張 feature map `Flatten` 後接大型 fully connected layer。
GAP 則對每個 channel 的所有空間位置取平均：

```text
[B, C, H, W] → AdaptiveAvgPool → [B, C, 1, 1] → [B, C]
```

### 想解決什麼

- 大幅減少 fully connected parameters。
- 降低模型記住特定像素位置的能力。
- 讓每個 channel 表示某類全域特徵是否出現。

### 結果

- GAP：Public 60.573%。
- GAP + pseudo-label：Public 64.277%。

這表示模型架構本身比早期 baseline 更適合此資料；再加入 unlabeled data
後繼續提升。但 GAP 會把所有空間位置平均，也可能丟失食物局部排列資訊。

---

## 10. 實驗 13～14：Residual Architecture

### 改用了什麼

Residual block 不只學 `F(x)`，而是輸出：

```text
y = ReLU(F(x) + shortcut(x))
```

Shortcut 讓原始特徵與 gradient 能更直接穿過多層網路。當 channel 或
feature map 大小改變時，使用 1×1 convolution 對齊形狀。

### 想解決什麼

普通深層 CNN 可能因路徑太長而難以 optimization。Residual connection
提供較短的 gradient path，使更深模型較容易學習。

### 結果

- Residual：Public 62.843%。
- Residual + pseudo-label 0.95：Public 64.695%。

Residual 單模型優於早期 baseline，但 pseudo-label 後與 GAP pseudo 很
接近。這代表架構有幫助，資料量仍是共同瓶頸。

---

## 11. 實驗 15～16：CutMix

### 改用了什麼

CutMix 從同一 batch 的另一張圖片剪下一塊矩形貼入目前圖片，並依實際
面積比例混合兩個 label 的 loss：

```text
loss = λ·CE(pred, label_A) + (1-λ)·CE(pred, label_B)
```

### 想解決什麼

- 增加局部遮擋與組合變化。
- 避免模型只靠一小塊最顯眼區域判斷。
- 讓 decision boundary 更平滑。

### 結果

40 epochs 的 Public 只有 62.246%，但訓練到 80 epochs 後提升到
65.710%。CutMix 把任務變難，因此短時間內看似退步；較長訓練才能
吸收這種 regularization。

這也說明：比較增強方法時應讓各模型充分收斂，不能只固定很短 epoch。

---

## 12. 實驗 17～19：Ensemble 與 TTA

### Ensemble 做了什麼

將 GAP pseudo、Residual pseudo 與 Residual CutMix 三個模型的 Softmax
機率加權平均，再取最大類別。權重由 validation 比較選出，例如：

```text
GAP 0.30 + Residual 0.40 + CutMix 0.30
```

不同架構可能犯不同錯誤；平均後可降低單一模型的 variance。三模型
Ensemble 最終 Public 為 68.100%。

### TTA 做了什麼

Test-Time Augmentation 對原圖與水平翻轉圖各推論一次，再平均機率。
它不更新模型，只增加推論成本。

TTA 版 Public 提升到 68.697%，但 Private 從 68.798% 降到 68.499%。
這表示水平翻轉只帶來很小且不一致的改變，不能視為穩定突破。

### 為什麼 Ensemble 仍無法到 70%

Ensemble 能消除模型彼此不同的錯誤，卻不能修正大家共同的限制。如果
三個模型都缺少足夠資料或高階特徵，平均答案也不會產生它們從未學到的
知識。它主要改善預測整合，不是提升每個模型的表示能力。

---

## 13. 最終實驗：Reference 五層 CNN + Dynamic Self-labeling

最後不再繼續堆疊技巧，而是依參考 notebook 建立獨立的
`hw03_reference.py`，完整重現一套一致的訓練 recipe。

### 13.1 更深、更寬的 CNN

```text
3 → 64 → 128 → 256 → 512 → 1024 channels
```

每一個 block 都是：

```text
Conv2d(kernel=3, no padding)
→ BatchNorm2d
→ ReLU
→ MaxPool2d(2)
```

輸入 128×128 經五層後得到 `[1024, 2, 2]`，flatten 成 4,096 個特徵，
再經過：

```text
4096 → 1024 → BN → ReLU → Dropout 0.6
     → 256  → BN → ReLU → Dropout 0.4
     → 11 logits
```

相較 baseline，這個模型能逐層建立更高階的表示：

```text
邊緣與顏色
→ 紋理與局部形狀
→ 食材組件
→ 食物整體外觀
→ 類別判斷
```

它的容量更強，但也更容易 overfit，所以必須搭配下面的強增強、Dropout
與更多資料。不能把提升只歸因於「層數變多」。

### 13.2 更強的資料增強

Reference training transform 包含：

- 不限制狹窄範圍的 `RandomResizedCrop(128×128)`。
- 在 default、CIFAR10、SVHN 三種 AutoAugment policy 中隨機選擇。
- `RandomHorizontalFlip(p=0.5)`。
- `ColorJitter(brightness=0.5)`。
- `RandomAffine(degrees=20, translate=0.2, scale=0.7~1.3)`。

它比前期的保守 augmentation 更強。模型每次看到的圖片更難，因此 train
accuracy 不一定高於 validation accuracy；這不代表程式錯誤，而是兩者
難度不同。

### 13.3 BatchNorm 與 Dropout 的角色

BatchNorm 穩定每層 activation 的數值範圍，使五層 CNN 較容易
optimization。大型 FC 使用 Dropout 0.6 / 0.4，抑制模型依賴固定少數
features。

這也解釋為什麼早期小模型的 Dropout 0.3 太強，但 Reference 的 0.6
仍可訓練：regularization 強度必須和模型容量、資料增強及 epoch 一起看。

### 13.4 Gradient Clipping

每個 training batch 在 `loss.backward()` 後計算 gradient norm，並限制：

```text
max gradient norm = 10
```

如果 gradient 太大，就依比例縮小，避免一次異常更新破壞模型。它主要
提供訓練穩定性，不是直接讓 accuracy 增加的分類技巧。

### 13.5 Dynamic Self-labeling

這是 Reference 與先前「單次 pseudo-label fine-tune」最大的差異。

```text
只使用 3,080 張 labeled data
             ↓
Best validation accuracy > 70%
             ↓
Teacher 推論 6,786 張 unlabeled data
             ↓
保留 Softmax confidence ≥ 0.90
             ↓
labeled + pseudo-labeled 合併訓練
             ↓
每 5 epochs 重新產生 pseudo-label
```

為什麼要等 validation 超過 70%？因為太弱的 teacher 會製造大量錯誤
標籤，接著把自己的錯誤再學回去，形成 confirmation bias。

為什麼 threshold 使用 0.90？它在標籤品質與涵蓋數量之間折衷。低信心
圖片先不用，避免把所有 unlabeled data 不加選擇地塞進 training set。

為什麼每 5 epochs 更新？teacher 進步後，原本低信心的圖片可能變得
可靠，部分舊預測也可能改變。動態更新形成：

```text
模型改善 → pseudo-label 更可靠／更多 → 有效資料增加 → 模型再改善
```

本次訓練在 epoch 190 首次啟動，加入 1,783 張 pseudo-labeled images；
後期約使用 4,287 張。合併只存在 Dataset/DataLoader 中，不會修改原始
圖片或永久寫入假標籤，因此是可逆的。

### 13.6 為什麼需要 500 epochs

這次不是把同一批簡單資料機械式重複 500 次：

- On-the-fly augmentation 每次產生不同輸入。
- Epoch 190 後 training set 加入並持續更新 pseudo-label。
- 深層模型與強 Dropout 需要較長時間收斂。

因此訓練可分成三階段：

1. **Epoch 1～189：監督學習**，先建立可靠 teacher。
2. **Epoch 190 後：半監督學習**，利用高信心 unlabeled data。
3. **後期：收斂平台**，valid accuracy 在 74%～76% 附近波動。

最佳點是 epoch 443，而非最後 epoch 500，所以程式保存 validation
accuracy 最佳的 checkpoint，再用它產生 submission。

---

## 14. 為什麼 Reference 最後明顯比較好

### 原因一：模型本身的表示能力更強

早期模型即使搭配 scheduler 或 Ensemble，單模型仍受限於特徵提取能力。
Reference 的五層 64→1024 CNN 能學到更複雜的 Food-11 特徵。

### 原因二：強增強與大模型彼此配合

只有大模型容易 overfit；只有強增強又可能 underfit。Reference 同時
提高模型容量與資料難度，讓模型有能力學，也必須學較穩健的規則。

### 原因三：真正擴大有效訓練資料

早期方法主要在 3,080 張 labeled images 上調參。Reference 後期約使用：

```text
3,080 labeled + 4,287 pseudo-labeled ≈ 7,367 images
```

這不是單純增加檔案副本，而是加入新的食物外觀、背景、光線與角度。

### 原因四：Pseudo-label 啟動時機較可靠

它不是一開始就相信模型，而是 valid accuracy 超過 70% 才啟動，並只
使用 confidence ≥ 0.90 的資料，比早期單次 pseudo-label 更有品質控制。

### 原因五：標籤會隨模型改善而更新

先前 pseudo-label 是一次性 fine-tune；Reference 每 5 epochs 更新，能
逐步擴充資料並修正 teacher 的判斷。

### 原因六：訓練 recipe 是一致的完整系統

前期實驗常一次加一個局部技巧，後期 Ensemble 則整合多個仍有限制的
模型。Reference 從架構、augmentation、regularization、optimization、
checkpoint 到 self-labeling 是互相配合的系統。

### 數據證據

```text
Baseline Public                 48.387%
Scheduler baseline Public       55.794%
三模型 Ensemble + TTA Public    68.697%
Reference Public                77.897%
```

Reference 相較初始 baseline 提升約 29.51 個百分點，相較最佳 Ensemble
再提升約 9.20 個百分點。它的 Validation 76.36%、Private 76.748% 與
Public 77.897% 彼此接近，表示提升不只出現在單一 leaderboard split。

---

## 15. 從 Training Curve 可以讀到什麼

最終曲線的紅線位於 epoch 443：

- Train loss 長期下降，表示 optimizer 持續找到更合適的參數。
- Validation loss 整體下降但波動較大，符合 validation 只有 660 張的情況。
- Epoch 190 左右曲線出現轉折，對應 pseudo-label 啟動、training set 改變。
- 後期 train/valid accuracy 約落在相近區域，沒有早期 baseline 巨大的
  overfitting gap。
- Epoch 443 後提升有限，因此 500 並非每一輪都帶來額外收益。

Reference 的 train accuracy 有時低於 valid accuracy，原因包括：

1. Training 使用更強的隨機 augmentation。
2. Training 啟用 Dropout，validation 會關閉。
3. Training 後期包含可能有雜訊的 pseudo-label。
4. Validation 使用乾淨 Resize 圖片，任務相對容易。

因此不能只看 train accuracy 大小判斷模型好壞，必須同時檢查資料處理與
evaluation mode。

---

## 16. 實驗設計上的限制

這次流程成功達到高分，但若要做更嚴謹的研究，仍有以下限制：

1. **沒有固定 random seed**：資料順序、初始化與 augmentation 不同，
   小幅分數差異可能只是隨機波動。
2. **部分實驗一次改多個變因**：例如架構、epoch 與 pseudo-label 同時
   改變，無法精確估計各自貢獻。
3. **只使用一個 validation split**：660 張偏少，估計存在抽樣誤差。
4. **反覆查看 Kaggle 分數**：若依 Public score 不斷選方法，可能對
   public leaderboard 過度調整。
5. **Pseudo-label 仍可能錯誤**：Softmax 高信心不保證一定正確，且類別
   分布不平衡時可能放大偏誤。
6. **500 epochs 成本高**：約 8 小時，之後應加入 early stopping 或先用
   小型消融實驗縮小候選範圍。

更嚴謹的下一次做法是固定 seed，建立相同 data split，每次只改一項，
至少重跑 3 次並回報平均值與標準差。

---

## 17. 最終應掌握的四個核心

### Model

CNN 不是答案本身，而是一組可學習的函數形式。深度、channel 與 pooling
決定它能表達什麼特徵。

### Data

模型只能從看過的資料分布學習。Augmentation 增加合理變化；pseudo-label
則嘗試把新的未標記樣本轉成可用訓練訊號。

### Loss 與 Optimization

Cross-entropy 衡量 logits 與類別答案的差距；backpropagation 計算
gradient；Adam 根據 gradient 更新參數。Scheduler、weight decay、
Dropout 與 gradient clipping 都是在控制「如何穩定地找到泛化較好的參數」。

### Evaluation

最終不能只看 train accuracy。應使用 validation 選 checkpoint，再用未
參與開發的 test/private split 確認泛化。這次三者接近，是結果可信的重要
證據。

---

## 18. 一句話總結

最後 Reference 版比較好，不是因為某個神奇參數，也不只是因為跑了
500 epochs，而是：

```text
更有能力的五層 CNN
+ 更強的 on-the-fly augmentation
+ BatchNorm / Dropout / gradient clipping
+ 先建立可靠 teacher
+ confidence threshold 控制 pseudo-label 雜訊
+ 每 5 epochs 動態更新未標記資料
+ 足夠的訓練時間
= 具備一致性的完整半監督學習 pipeline
```

這套 pipeline 最終達到 Validation 76.36%、Kaggle Private 76.748%、
Public 77.897%，並且比單純堆疊 Ensemble 與 TTA 更有效。
