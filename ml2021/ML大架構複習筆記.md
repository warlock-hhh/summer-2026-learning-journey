# ML 大架構複習筆記

> 範圍：目前 `ml2021/筆記` 中的基本概念、Deep Learning 訓練指引、優化技巧、分類 Loss、CNN 與 Self-Attention。
>
> 複習原則：先判斷「它屬於哪一類、位於哪一階段」，再理解「同類有哪些選擇、上下游接什麼、為什麼選它」。

---

## 0. 一句話總覽

機器學習是在一組候選函數中，利用訓練資料定義 Loss，再透過 Optimization 找出能讓 Loss 較小的參數；CNN 與 Self-Attention 是針對不同資料關係設計的模型架構，而 ReLU、Softmax、Cross-Entropy、Adam、Batch Normalization 等技術則分別負責模型內部運算、輸出、誤差衡量、參數更新與穩定訓練。

---

## 1. 機器學習的完整生命週期

```text
定義任務
  ├─ Regression：輸出連續數值
  ├─ Classification：輸出類別
  └─ Structured Learning：輸出具有結構的物件或序列
        ↓
準備資料
  ├─ Training set：學習參數
  ├─ Validation set：選模型與超參數
  └─ Testing set：最後評估泛化能力
        ↓
選擇模型／候選函數集合
  ├─ Linear model / Fully Connected Network
  ├─ CNN：利用影像的局部性與參數共享
  ├─ Self-Attention：建立序列中各位置的關係
  └─ Transformer：以 Attention Blocks 組成序列理解與生成系統
        ↓
Forward Pass
  輸入 → 線性運算／特徵抽取 → Activation → 輸出 logits
        ↓
輸出轉換
  ├─ Binary / Multi-label：Sigmoid
  └─ Multi-class single-label：Softmax
        ↓
Loss
  預測與標籤比較，得到一個可優化的純量
        ↓
Backpropagation
  使用 Chain Rule 計算每個參數的 gradient
        ↓
Optimizer Update
  SGD / Momentum / Adagrad / RMSProp / Adam
        ↓
重複 Batch → 完成 Epoch → Validation
        ↓
診斷 Model Bias、Optimization、Overfitting、Mismatch
        ↓
最後 Testing / Inference
```

### 一次參數更新的最小閉環

```text
一個 Batch
  ↓
Model Forward
  ↓
Prediction / Logits
  ↓
Loss Function
  ↓
Backward：計算 Gradient
  ↓
Optimizer：更新 Weight 與 Bias
  ↓
下一個 Batch
```

名詞對應：

- **Batch**：一次拿來計算 Loss 與 Gradient 的一小組資料。
- **Update / Step**：Optimizer 更新一次參數。
- **Epoch**：所有 training data 都被看過一次。
- **Shuffle**：每個 Epoch 前重新打亂資料，讓 Batch 組合改變。
- **Hyperparameter**：不是直接由梯度學出的設定，例如 learning rate、batch size、kernel size、stride、head 數。
- **Parameter**：模型真正學習的值，例如 weight、bias、filter，以及 Q/K/V projection matrix。

---

## 2. 技術分類總樹

```text
Machine Learning
│
├─ A. 任務類型
│  ├─ Regression
│  ├─ Classification
│  └─ Structured Learning / Sequence-to-Sequence
│
├─ B. 模型架構（資料如何轉成預測）
│  ├─ Linear Model / Fully Connected Network
│  ├─ CNN
│  ├─ Self-Attention / Multi-head Self-Attention
│  └─ Transformer
│     ├─ Encoder-only
│     ├─ Decoder-only
│     └─ Encoder-Decoder
│
├─ C. 模型內部元件
│  ├─ Linear / Fully Connected Layer
│  ├─ Convolution / Filter
│  ├─ Pooling
│  ├─ Attention Layer
│  ├─ Feed-Forward Network
│  ├─ Residual Connection
│  ├─ Layer Normalization
│  └─ Activation Function
│     ├─ Sigmoid
│     └─ ReLU
│
├─ D. 輸出表示
│  ├─ Raw value（Regression）
│  ├─ Sigmoid（Binary / Multi-label）
│  └─ Softmax（Multi-class single-label）
│
├─ E. Loss Function（如何衡量錯誤）
│  ├─ MAE
│  ├─ MSE
│  ├─ Binary Cross-Entropy
│  └─ Categorical Cross-Entropy
│
├─ F. 梯度與參數更新
│  ├─ Backpropagation / Chain Rule
│  ├─ Vanilla Gradient Descent / SGD
│  ├─ Momentum
│  ├─ Adagrad
│  ├─ RMSProp
│  └─ Adam = RMSProp + Momentum 的核心想法
│
├─ G. 訓練控制與穩定化
│  ├─ Batch Size
│  ├─ Learning Rate
│  ├─ Learning Rate Decay
│  ├─ Warm Up
│  └─ Normalization
│     ├─ Batch Normalization
│     ├─ Layer Normalization
│     ├─ Instance Normalization
│     └─ Group Normalization 等
│
├─ H. 泛化與避免 Overfitting
│  ├─ Data Augmentation
│  ├─ Early Stopping
│  ├─ Regularization
│  └─ Dropout
│
└─ I. 評估與模型選擇
   ├─ Training / Validation / Testing
   ├─ Cross Validation
   ├─ N-fold Cross Validation
   └─ Mismatch 檢查
```

---

## 3. 同一階段有哪些不同選擇？

### 3.1 任務類型：模型到底要輸出什麼？

| 類型 | 輸出 | 例子 | 關鍵差異 |
|---|---|---|---|
| Regression | 連續數值 | 溫度、房價 | 答案是數值大小 |
| Classification | 離散類別 | 貓／狗、Food-11 | 答案是類別 |
| Structured Learning | 結構化結果 | 翻譯句子、序列標記 | 輸出元素間具有結構關係 |

### 3.2 Activation Function：在線性層後加入非線性

共同位置：

```text
輸入 x → z = Wx + b → Activation f(z) → 下一層
```

| 方法 | 所屬類別 | 特色 | 常見用途 | 主要限制 |
|---|---|---|---|---|
| Sigmoid | Activation Function | 輸出介於 0 與 1，平滑 | 傳統隱藏層、二元分類輸出 | 飽和區梯度小，深層隱藏層較難訓練 |
| ReLU | Activation Function | `max(0,z)`，正區間梯度直接 | 深層網路與 CNN 隱藏層 | 負區間梯度為 0 |

重要觀念：若沒有 Activation，多層線性運算仍可合併成一個線性運算，深度無法帶來真正的非線性表示能力。

### 3.3 輸出轉換：把 logits 變成適合任務的表示

| 任務 | Output Activation | Loss | 原因 |
|---|---|---|---|
| 二元分類 | Sigmoid | Binary Cross-Entropy | 每筆資料估計一個正類機率 |
| 多類別、單標籤 | Softmax | Categorical Cross-Entropy | 所有類別互斥，機率總和為 1 |
| 多類別、多標籤 | 每類各自 Sigmoid | Binary Cross-Entropy | 各標籤可以同時成立，不應互相競爭 |
| 一般 Regression | 常不加輸出 Activation | MAE / MSE | 輸出不必限制在 0 到 1 |
| 0 到 1 的 Regression | Sigmoid | MSE 或 BCE，依問題定義 | 需要限制輸出範圍 |

注意：**Logit 是 Softmax 或 Sigmoid 之前的原始分數，不是機率。**

### 3.4 Loss Function：同一預測可用不同方式衡量錯誤

| Loss | 適合任務 | 它在量什麼 | 技術重點 |
|---|---|---|---|
| MAE | Regression | 絕對誤差平均 | 對極端誤差不像 MSE 那麼敏感 |
| MSE | Regression | 平方誤差平均 | 大錯誤會被平方放大 |
| Binary Cross-Entropy | 二元／多標籤分類 | 每個二元事件的機率差異 | 通常配 Sigmoid |
| Categorical Cross-Entropy | 多類別單標籤 | 正確類別的預測機率是否夠高 | 通常配 Softmax |

教材的重要比較：分類若使用 MSE，某些「錯得很嚴重」的位置可能梯度仍很平；Cross-Entropy 通常能提供較適合分類優化的梯度。因此 Loss 不只定義評分方式，也會改變訓練路徑。

### 3.5 Optimizer：都是更新參數，但記憶資訊不同

共同位置：

```text
Loss → Backprop 得到 Gradient → Optimizer 決定實際更新方向與步幅
```

| 方法 | 使用的主要資訊 | 直覺 | 適合用來理解的比較 |
|---|---|---|---|
| Vanilla GD / SGD | 當前 gradient | 沿現在的下坡方向走 | 基準方法 |
| Momentum | 當前 gradient + 過去更新方向 | 像帶有慣性的球，減少來回震盪 | 與 SGD 比較「有沒有歷史方向」 |
| Adagrad | 累積歷史 gradient 平方 | 每個參數有自己的有效 learning rate | 累積值只增不減，步伐可能越來越小 |
| RMSProp | gradient 平方的移動平均 | 保留近期尺度，改善 Adagrad 持續縮小問題 | 與 Adagrad 比較「全部歷史 vs 近期歷史」 |
| Adam | Momentum + RMSProp 核心概念 | 同時利用方向慣性與自適應步幅 | 常用的通用起點 |

### 3.6 Learning Rate 策略：控制「每一步走多遠」

| 策略 | 所屬類別 | 生命週期位置 | 目的 |
|---|---|---|---|
| 固定 Learning Rate | 訓練超參數 | 每次 optimizer update | 簡單，但全程使用同一步幅 |
| Learning Rate Decay | LR Scheduling | 訓練中後期 | 越接近較佳區域，步伐越小 |
| Warm Up | LR Scheduling | 訓練初期 | 先小後大，使早期統計與方向尚不穩定時不要走太遠 |

判讀：Loss 不下降不一定是 gradient 為 0，也可能是 learning rate 太大，導致在狹窄山谷兩側震盪。

### 3.7 Normalization：都是調整分布，但統計維度不同

| 方法 | 主要統計範圍 | 重點 |
|---|---|---|
| Feature Normalization | 輸入資料各 feature | 讓不同維度尺度接近，改善 error surface |
| Batch Normalization | 一個 batch 內同一 feature/channel | 訓練時計算 batch 統計；推論時使用訓練期間累積的統計量 |
| Layer Normalization | 同一筆資料的一層 feature | 不依賴 batch 統計，常與序列模型搭配；本教材列為其他 Normalization 方法 |
| Instance / Group Normalization | 單一樣本或 channel group | 對 batch size 較不敏感的其他選項 |

Batch Normalization 常見串接：

```text
Linear / Convolution → Batch Normalization → Activation
```

教材指出 Normalization 放在 Activation 前後都可討論；若使用 Sigmoid，將輸入移到接近 0 的高斜率區域，有利於 Gradient 傳遞。

### 3.8 避免 Overfitting：目標相同，介入位置不同

| 技術 | 介入位置 | 核心作法 |
|---|---|---|
| Data Augmentation | 資料端 | 產生合理變形，增加資料多樣性 |
| Regularization | Loss / 參數約束 | 限制模型使用過度複雜的參數 |
| Dropout | 模型訓練端 | 訓練時隨機停用部分神經元 |
| Early Stopping | 訓練控制端 | Validation 表現不再改善時停止 |

這四項不是同一個算子，但都屬於「改善泛化、降低 Overfitting」的大類。

---

## 4. 訓練失敗時，先判斷是哪一類問題

### 4.1 診斷生命週期

```text
Training Loss 高
  ├─ Model Bias：候選函數集合根本表達不了目標
  │    → 換更有能力的模型、增加 feature 或模型複雜度
  │
  └─ Optimization Issue：模型理論上可以，但參數沒有找到
       → 檢查 learning rate、optimizer、batch size、normalization

Training Loss 低，但 Validation / Testing Loss 高
  └─ Overfitting
       → 更多資料、augmentation、regularization、dropout、early stopping

Validation 好，但真實 Testing 差
  ├─ 反覆看 validation/public test，造成對該集合 overfit
  └─ Mismatch：訓練與真實資料分布不同
```

### 4.2 Model Bias 與 Optimization 的區分

- 換成更深或更複雜的模型後，**training loss 仍沒有比淺模型低**：先懷疑 Optimization。
- 複雜模型確實能把 training loss 壓得更低，但 validation 變差：先懷疑 Overfitting。
- 模型複雜度增加會降低 Model Bias，卻可能提高 Overfitting 風險，這就是 Bias-Complexity Trade-off。

### 4.3 Critical Point 的分類

```text
Critical Point：Gradient = 0
  ├─ Local Minimum：周圍方向都上升
  ├─ Local Maximum：周圍方向都下降
  └─ Saddle Point：有些方向上升、有些方向下降
```

Hessian 特徵值的概念判斷：

- 全正：Local Minimum。
- 全負：Local Maximum。
- 有正有負：Saddle Point。

高維空間中 Saddle Point 通常比真正的 Local Minimum 常見。較小 Batch 帶來的 noisy gradient，以及 Momentum 的歷史方向，都可能幫助離開困難區域。

### 4.4 Small Batch 與 Large Batch

| 面向 | Small Batch | Large Batch |
|---|---|---|
| Gradient | 較 noisy | 較穩定、接近全資料 gradient |
| 每個 Epoch 的 update 次數 | 多 | 少 |
| 單次 update 計算 | 較少 | 較多，但 GPU 可平行化 |
| 泛化直覺 | Noise 可能幫助走向較平坦區域 | 可能較容易走入尖銳 minima |
| 限制 | 計算吞吐可能較差、BN 統計較不穩 | 記憶體需求高，泛化可能受影響 |

Batch Size 同時是**計算資源、優化路徑與泛化**的超參數，不只是「一次塞多少資料」。

---

## 5. 模型架構的層級關係

### 5.1 不要把「模型、Layer、函數、訓練技巧」放在同一層

| 層級 | 例子 | 它回答的問題 |
|---|---|---|
| 任務 | Classification | 我要預測什麼？ |
| 模型架構 | CNN、Self-Attention | 資料關係如何被建模？ |
| Layer / 元件 | Convolution、FC、Attention、Pooling | 模型內部有哪些處理單元？ |
| Activation | ReLU、Sigmoid | 線性運算後如何加入非線性？ |
| Output Transformation | Sigmoid、Softmax | logits 如何變成任務需要的表示？ |
| Loss | MSE、Cross-Entropy | 怎麼量化錯誤？ |
| Optimizer | SGD、Adam | 怎麼更新參數？ |
| 訓練技巧 | BN、Dropout、LR Schedule | 怎麼穩定訓練或改善泛化？ |

### 5.2 同一名詞可能扮演不同角色

**Sigmoid**：

- 放在 hidden layer：身分是 Activation Function，用來加入非線性。
- 放在 binary 或 multi-label output：身分是 Output Transformation，用來得到各事件的 0 到 1 分數。

**Softmax**：

- 數學上也是非線性函數。
- 在目前教材架構裡，最重要的定位是「多類別單標籤的輸出轉換」，不是一般 hidden layer activation。

**Batch Normalization**：

- 不是 Activation，也不是 Optimizer。
- 它是訓練穩定化／Normalization 技術，調整中間特徵分布。

**Pooling**：

- 沒有需要梯度學習的參數，是下採樣 operator。
- 它不是 Activation，也不是 Convolution Filter；現代架構也可能用 stride convolution 取代它。

---

## 6. CNN：影像模型的分類與生命週期

### 6.1 CNN 在整體架構中的位置

CNN 屬於**模型架構／特徵抽取器**。它不是 Loss，也不是 Optimizer。它先把影像轉成有用的 feature，再交給分類頭產生 logits。

```text
Image
  ↓
Convolution：Filter 偵測局部 pattern
  ↓
Activation：例如 ReLU
  ↓
Pooling 或 Strided Convolution：降低空間尺寸
  ↓
重複數次，逐層組合更高階 pattern
  ↓
Flatten / Global representation
  ↓
Fully Connected
  ↓
Logits → Softmax / Cross-Entropy
```

### 6.2 CNN 的兩個核心先驗

| 核心概念 | 它限制了什麼 | 帶來的好處 |
|---|---|---|
| Receptive Field | 神經元只看局部範圍 | 利用影像 pattern 通常是局部的特性 |
| Parameter Sharing | 同一 Filter 掃過不同位置並共用權重 | 大幅減少參數，讓同一 pattern 可在不同位置被偵測 |

這些限制增加 CNN 的 Model Bias，但這是針對影像設計的有用偏好；相較 Fully Connected Network，通常更省參數、較不易 Overfit。

### 6.3 Convolution 同一階段的設定

| 名稱 | 所屬類別 | 控制什麼 |
|---|---|---|
| Kernel / Filter Size | Convolution 超參數 | 一次看的局部範圍 |
| Number of Filters | Convolution 超參數 | 輸出 feature map 的 channel 數 |
| Stride | Convolution 超參數 | Filter 每次移動距離，影響輸出空間尺寸 |
| Padding | 邊界處理 | 是否保留邊緣資訊及控制輸出尺寸 |
| Pooling Size / Type | 下採樣設定 | 如何縮小 feature map |

### 6.4 Filter 與 Feature Map 的關係

```text
一個 Filter + 整張輸入影像
        ↓ 掃描並做局部內積
一個輸出 Feature Map / Channel

多個 Filters
        ↓
多個輸出 Channels
```

前層 Filter 常偵測簡單 pattern；疊加多層後，後層神經元的實際 receptive field 會擴大，能組合成更複雜的 pattern。

### 6.5 Pooling 的同類方法

| 方法 | 做法 | 特性 |
|---|---|---|
| Max Pooling | 區域內取最大值 | 保留最強烈的 pattern response |
| Mean Pooling | 區域內取平均值 | 保留區域平均資訊 |

Pooling 可有可無。它能降低空間尺寸與計算量，但也會丟失資訊；教材指出現代網路可能減少 Pooling，改用全 Convolution 的設計。

### 6.6 CNN 的限制與對應技術

- CNN 本身不天然保證對縮放與旋轉穩健。
- 可用 Data Augmentation 將旋轉、裁切、縮放等合理變化加入訓練資料。
- 因此 Data Augmentation 屬於資料／泛化技巧，不是 CNN Layer。

---

## 7. Self-Attention：序列關係模型的分類與生命週期

### 7.1 它在整體架構中的位置

Self-Attention 屬於**模型架構中的關係建模元件**。它接收一串向量，讓每個位置依內容決定要參考序列中的哪些位置，再輸出同樣是一串、但已融合上下文的向量。

```text
Input Sequence：a1, a2, ..., an
        ↓
每個 ai 經線性投影產生 qi、ki、vi
        ↓
Query 與所有 Key 計算相關分數
        ↓
Scaling + Softmax 得到 Attention Weights
        ↓
以權重加總所有 Value
        ↓
Context-aware Output：b1, b2, ..., bn
```

核心式：

```text
Attention(Q, K, V) = softmax(QKᵀ / √dk)V
```

### 7.2 Q、K、V 的角色分類

| 元件 | 問的問題 | 直覺 |
|---|---|---|
| Query | 我現在要找什麼資訊？ | 當前位置提出的查詢 |
| Key | 我擁有什麼可供比對的特徵？ | 每個位置的索引／標籤 |
| Value | 如果關注我，要拿走什麼內容？ | 真正被加權匯總的資訊 |

Q、K、V 都是由輸入向量透過**不同、可學習的線性投影**產生。它們不是三份不同的原始資料。

### 7.3 Attention 流程中的技術歸類

| 技術 | 類別 | 在 Attention 中的角色 |
|---|---|---|
| Dot-product / Additive | 相似度計算方法 | 計算 Query 與 Key 的相關程度 |
| Scaling `1/√dk` | 數值穩定技巧 | 避免向量維度大時內積過大 |
| Softmax | 權重正規化 | 將相關分數轉成總和為 1 的權重 |
| Weighted Sum | 資訊聚合 | 使用權重組合所有 Value |

注意：此處 Softmax 的角色是「將 attention scores 正規化成權重」，不是最終分類輸出。

### 7.4 Multi-head Self-Attention

```text
同一組輸入
  ├─ Head 1：學一種關係
  ├─ Head 2：學另一種關係
  └─ Head h：學其他關係
        ↓
Concat
        ↓
Linear Projection
```

Multi-head 仍屬於 Self-Attention 類別，不是獨立的 Loss 或 Optimizer。Head 數是超參數；各 Head 使用不同的 Q/K/V 投影，使模型能在不同表示子空間中關注不同關係。

### 7.5 Positional Encoding

純 Self-Attention 只依內容計算關係，沒有自然的先後順序概念。因此要把位置資訊加入輸入：

```text
Token / Input Embedding + Positional Encoding
                    ↓
              Self-Attention
```

Positional Encoding 屬於**輸入表示技術**，不是 Attention Score，也不是 Activation。

### 7.6 CNN、RNN、Self-Attention 與 Transformer 的比較

前三者描述主要的關係建模方式；Transformer 則是以 Attention 為核心組成的完整架構：

| 架構 | 元素如何取得上下文 | 優點 | 主要限制／代價 |
|---|---|---|---|
| CNN | 固定局部 receptive field，靠堆疊擴大範圍 | 影像局部先驗強、參數共享、資料效率較好 | 長距離關係需多層傳遞 |
| RNN | 按順序把歷史濃縮進 hidden state | 自然表示順序 | 難平行化，長距離資訊需逐步傳遞 |
| Self-Attention | 每個位置直接和所有位置比較 | 關係彈性高、可平行、長距離直接連接 | 計算量隨序列長度快速增加，通常需要較多資料 |
| Transformer | 堆疊 Attention、FFN、Residual 與 LayerNorm | 可組成理解、生成或 Seq2Seq 系統 | 架構較複雜，Attention 的長序列成本仍高 |

概念對應：Self-Attention 的全域關注可視為「由內容動態決定的 receptive field」；CNN 的 receptive field 則主要由 kernel、stride 與堆疊深度預先決定。

### 7.7 任務型態與輸出數量

| 型態 | 輸入／輸出關係 | 例子 |
|---|---|---|
| Sequence Labeling | 每個輸入位置都有一個輸出 | POS tagging |
| Sequence Classification | 一串輸入得到一個分類結果 | 情感分類 |
| Sequence-to-Sequence | 一串輸入產生另一串輸出，長度可不同 | 語音辨識、翻譯 |

Self-Attention 主要解決「序列元素彼此如何交換資訊」，最後輸出頭仍要依任務選擇分類或序列生成方式。

---

## 8. Transformer：完整 Seq2Seq 系統

### 8.1 Transformer 與 Self-Attention 的層級差異

```text
Self-Attention
  = 一種讓序列內各位置交換資訊的運算元件

Transformer Block
  = Attention + Feed-Forward Network + Residual Connection + Layer Normalization

Transformer
  = 堆疊多個 Encoder／Decoder Block，完成理解、生成或 Seq2Seq 任務的模型架構
```

| 名稱 | 所屬層級 | 輸入／輸出 | 主要責任 |
|---|---|---|---|
| Self-Attention | 模型元件 | 一串向量 → 一串上下文化向量 | 建立同一序列內的位置關係 |
| Transformer Encoder Block | 複合模組 | 一串向量 → 一串更高階表示 | Self-Attention 後再逐位置做 Feed-Forward 處理 |
| Transformer Encoder | 模型子系統 | 原始輸入序列 → context representations | 理解並編碼整段輸入 |
| Transformer Decoder | 模型子系統 | 已生成 token + Encoder 表示 → 下一 token 分布 | 根據輸入內容逐步生成輸出 |
| Transformer | 完整模型架構 | 一段序列 → 一段表示或另一段序列 | 組合 Encoder、Decoder 或其中一側完成任務 |

關鍵結論：**Self-Attention 是 Transformer 的核心元件，但 Transformer 不只等於 Self-Attention。**

### 8.2 Seq2Seq 在任務生命週期的位置

Seq2Seq（Sequence-to-Sequence）是「輸入與輸出都是序列，而且兩者長度不一定相同」的任務框架，可用於語音辨識、翻譯、語音合成、摘要與問答。

```text
Input Sequence（長度 N）
        ↓
Encoder：理解整段輸入
        ↓
Context Representations
        ↓
Decoder：依條件產生輸出
        ↓
Output Sequence（長度 N'，可與 N 不同）
```

Seq2Seq 是**任務／輸入輸出框架**；Transformer 是可實現 Seq2Seq 的**模型架構**。Seq2Seq 也可以由 RNN 等其他架構實現。

### 8.3 Transformer Encoder 的生命週期

```text
Token Embedding + Positional Encoding
        ↓
Multi-Head Self-Attention
        ↓
Residual Connection + Layer Normalization
        ↓
Position-wise Feed-Forward Network
        ↓
Residual Connection + Layer Normalization
        ↓
重複 N 個 Encoder Blocks
        ↓
每個輸入位置的 Context Representation
```

各元件的責任：

| 元件 | 所屬類別 | 它解決的問題 | 如果沒有它 |
|---|---|---|---|
| Positional Encoding | 輸入表示 | 補上 token 的順序 | Self-Attention 無法自然分辨位置先後 |
| Multi-Head Self-Attention | 關係建模 | 讓每個位置讀取其他位置 | 每個位置難以取得全域上下文 |
| Residual Connection | 訊息／梯度捷徑 | 保留原始表示並協助深層梯度傳遞 | 深層堆疊較難訓練，也可能遺失原資訊 |
| Layer Normalization | 訓練穩定化 | 對單一 token 的 feature 做正規化 | 中間表示尺度較不穩定 |
| Feed-Forward Network | 特徵轉換 | 對每個位置獨立做非線性轉換 | 只有位置間混合，缺乏充分的逐位置特徵處理 |

### 8.4 Self-Attention、Masked Self-Attention、Cross-Attention

三者都屬於 Attention，但 **Q、K、V 的來源與可見範圍不同**：

| 方法 | Query 來源 | Key／Value 來源 | 可看範圍 | 主要位置 |
|---|---|---|---|---|
| Encoder Self-Attention | Encoder 當前序列 | 同一 Encoder 序列 | 整段輸入 | Encoder |
| Masked Self-Attention | Decoder 已有序列 | 同一 Decoder 序列 | 只能看自己與之前位置 | Autoregressive Decoder |
| Cross-Attention | Decoder 中間表示 | Encoder 輸出 | 全部輸入位置 | Encoder-Decoder 之間 |

```text
Encoder Self-Attention：輸入自己和自己溝通
Decoder Masked Self-Attention：輸出前綴自己和自己溝通，但不能偷看未來
Cross-Attention：Decoder 提問，Encoder 提供可查詢的輸入內容
```

Masked Self-Attention 的遮罩會把未來位置的 Attention Score 設為不可選，使第 `t` 個位置只能依賴 `1...t`。這是避免訓練時洩漏未來答案的**因果限制**。

### 8.5 Transformer Decoder 的生命週期

```text
<BOS> + 已生成的 Tokens
        ↓
Embedding + Positional Encoding
        ↓
Masked Multi-Head Self-Attention
        ↓
Residual + LayerNorm
        ↓
Cross-Attention（Q 來自 Decoder；K/V 來自 Encoder）
        ↓
Residual + LayerNorm
        ↓
Feed-Forward Network
        ↓
Residual + LayerNorm
        ↓
Linear → Vocabulary Logits → Softmax
        ↓
選出下一個 Token，直到 <EOS>
```

Decoder 的輸出維度通常等於 vocabulary size。Softmax 將每個 token 的 logits 轉成詞彙機率，生成策略再決定實際選哪個 token。

### 8.6 Encoder-only、Decoder-only、Encoder-Decoder

| 架構 | 保留部分 | 資訊流 | 適合任務 | 直覺 |
|---|---|---|---|---|
| Encoder-only | Encoder | 可雙向讀取整段輸入 | 分類、序列標記、表示學習 | 重點是「理解」 |
| Decoder-only | Autoregressive Decoder | 只能依賴已知前文 | 文字生成、語言模型 | 重點是「續寫」 |
| Encoder-Decoder | Encoder + Decoder + Cross-Attention | 先理解輸入，再條件式生成 | 翻譯、摘要、語音辨識 | 重點是「輸入轉輸出」 |

這三種是 Transformer 架構層級的不同組合，不是三種 Activation 或 Optimizer。

### 8.7 Autoregressive（AT）與 Non-Autoregressive（NAT）

| 面向 | Autoregressive Decoder | Non-Autoregressive Decoder |
|---|---|---|
| 生成方式 | 一次生成一個 token | 多個位置平行生成 |
| 依賴關係 | 後一 token 依賴先前生成結果 | 降低或移除輸出位置間的順序依賴 |
| 推論速度 | 慢，無法完全平行 | 快，可高度平行 |
| 輸出長度 | 遇到 `<EOS>` 自然停止 | 通常需要先預測長度或準備指定數量的輸入位置 |
| 表現難點 | 錯誤會沿序列累積 | Multi-modality：同一輸入可能有多種合理輸出，平行決策較難協調 |
| 通常品質 | 較好 | 教材指出通常低於 AT，但速度較快 |

選擇邏輯：重視生成品質與條件依賴時偏向 AT；對延遲非常敏感且能接受品質取捨時才考慮 NAT。

### 8.8 Teacher Forcing、Free Running、Scheduled Sampling

這三者是 **Decoder 訓練／推論時，下一步餵入什麼 token** 的不同策略：

| 方法 | 下一步輸入 | 使用階段 | 優點 | 問題 |
|---|---|---|---|---|
| Teacher Forcing | Ground Truth token | 訓練 | 訓練穩定、每一步都有正確歷史 | 與推論情境不同，造成 Exposure Bias |
| Free Running | 模型自己上一步的預測 | 推論；也可用於訓練 | 與真實生成情境一致 | 早期錯誤可能連鎖擴大，訓練較難 |
| Scheduled Sampling | Ground Truth 與模型預測按機率混用 | 訓練 | 嘗試縮小訓練與推論差距 | 混合比例是額外策略，且不保證完全解決問題 |

**Exposure Bias**：訓練時 Decoder 總看到正確前文，推論時卻只能看到自己可能出錯的前文，因此模型沒有充分學過「如何從自己的錯誤恢復」。

### 8.9 Greedy Decoding 與 Beam Search

| 面向 | Greedy Decoding | Beam Search |
|---|---|---|
| 每一步保留 | 當下機率最高的 1 個候選 | 累積分數最高的 `beam width` 個候選 |
| 計算與記憶體 | 低 | 較高，隨 beam width 增加 |
| 搜尋品質 | 容易被早期局部最佳選擇限制 | 有機會找到整體分數較高的序列 |
| 是否保證全域最佳 | 否 | 否，只是較廣的近似搜尋 |
| 超參數 | 幾乎沒有 | Beam width、長度正規化等 |

Beam Search 是**推論搜尋策略**，不屬於 Decoder Layer、Loss 或 Optimizer；較大的 beam 也不必然產生人類感受更好的答案。

### 8.10 Copy Mechanism、Guided Attention、Noise

| 技術 | 所屬類別 | 解決的問題 | 核心方法 |
|---|---|---|---|
| Copy Mechanism | 輸出機制 | 專有名詞或輸入中罕見詞難由固定 vocabulary 生成 | 允許 Decoder 直接複製輸入 token |
| Guided Attention | Attention 約束 | 已知輸入輸出應遵循特定對齊模式 | 對 Attention Matrix 加入對齊引導 |
| Monotonic / Location-aware Attention | Attention 變體 | 語音等任務常具有近似單調的時間對齊 | 限制或利用先前位置，引導關注順序 |
| Training Noise | 資料／正則化技巧 | 提升 Decoder 對不完美輸入的耐受度 | 訓練時刻意破壞部分 Decoder 輸入 |

這些技術位於不同層級：Copy 改變輸出來源；Guided Attention 改變對齊學習；Noise 改變訓練資料。它們不能當成同一插槽的完全替代品。

---

## 9. HW5 實作對照：英中 LSTM Seq2Seq + Attention

> 本節把課堂 Transformer 理論對應到目前真正完成的 HW5。重要邊界：HW5 屬於 Transformer／Seq2Seq 主題，但目前成功訓練的是 **LSTM Encoder-Decoder + Attention baseline**，不是完整 Transformer。

### 9.1 HW5 在大架構中的位置

```text
任務層：英文序列 → 繁體中文序列（Machine Translation / Seq2Seq）
        ↓
資料層：平行語料 → SentencePiece → Token ID → fairseq Binary → Batch
        ↓
模型層：Embedding → LSTM Encoder → Attention → LSTM Decoder → Token Logits
        ↓
學習層：Teacher Forcing → Label-Smoothed Cross-Entropy → Backprop → Adam
        ↓
推論層：Checkpoint → Beam Search → 中文翻譯
        ↓
評估層：Validation Loss + 中文 BLEU + 人工抽查
```

一句話記憶：**Encoder 負責讀，Attention 負責找，Decoder 負責寫。**

### 9.2 四個生命週期

| 生命週期 | 輸入 | 核心處理 | 輸出 |
|---|---|---|---|
| 資料 | 英中句對 | 清理、切分、Subword、數值化、Batching | GPU 可處理的 Token ID 張量 |
| 模型 | 英文 Token ID | Embedding、Encoder、Attention、Decoder | 每個中文位置的 8,000 個 logits |
| 學習 | Logits + 正確中文 Token | Loss、Backward、Gradient Clip、Adam | 更新後的模型參數 |
| 使用與評估 | Checkpoint + 英文句子 | Beam Search、Detokenization、BLEU | 翻譯結果與品質指標 |

Docker、WSL2、CUDA 與 fairseq 位於**工程執行層**，負責讓上述生命週期可執行與重現；它們不是 Seq2Seq 模型元件。

### 9.3 資料生命週期

```text
英中平行語料
→ 逐行對齊與清理
→ Train / Validation / Test
→ SentencePiece Subword
→ Token ID
→ fairseq .bin / .idx
→ Padding / Batch
```

實際資料切分：

| Split | 數量 | 用途 |
|---|---:|---|
| Train | 390,041 | 計算 Gradient、更新參數 |
| Validation | 3,939 | 選 Checkpoint、計算 Loss 與 BLEU |
| Test | 4,000 | 只能產生翻譯；中文 reference 是佔位符 |

#### Word、Character、Subword 的差異

| 切詞方式 | 優點 | 缺點 | HW5 選擇 |
|---|---|---|---|
| Word | 序列較短、單位直觀 | Vocabulary 大，罕見詞與未知詞問題嚴重 | 未使用 |
| Character | Vocabulary 小，幾乎沒有未知字 | 序列長，單位語意較弱 | 未使用 |
| Subword | 詞彙量與序列長度的折衷，可拆解罕見詞 | 切分不一定符合人類詞界 | SentencePiece，vocabulary 8,000 |

#### SentencePiece 與 fairseq-preprocess

| 工具 | 負責什麼 | 不負責什麼 |
|---|---|---|
| SentencePiece | 決定字串如何切成 Subword，並映射為 ID | 不負責訓練 Encoder／Decoder |
| fairseq-preprocess | 建字典，將 Token ID 整理成 `.bin/.idx` 高速資料集 | 不改變 Seq2Seq 的理論 |
| fairseq-train | 組織模型、Loss、Optimizer、Checkpoint 訓練流程 | 不取代對 Attention／LSTM 原理的理解 |

`--joined-dictionary` 表示英中兩側共用 Token 字典，**不等於**來源 Embedding、目標 Embedding與輸出權重必然共用。

### 9.4 Embedding：Token ID 到連續向量

```text
Token ID：[B, T]
        ↓ 查表
Embedding：[B, T, 256]
```

目前 Embedding matrix 為 `[8,000, 256]`。Token ID 只是索引；Embedding 才是可學習參數，會隨 Backpropagation 更新。

| 表示方法 | 維度與性質 | 是否可學 | 主要差異 |
|---|---|---|---|
| Token ID | 一個整數 | 否 | 只有索引意義，數字大小不代表語意距離 |
| One-hot | Vocabulary 維稀疏向量 | 否 | 能區分 Token，但高維且沒有相似度結構 |
| Embedding | 256 維稠密向量 | 是 | 緊湊，可從使用情境學到表示 |

### 9.5 目前 Baseline 的模型資料流

```text
英文 Token ID
→ 256-d Embedding
→ 1-layer LSTM Encoder（hidden size 512）
→ 所有 Encoder Hidden States
→ LSTM Encoder-Decoder Attention
→ 1-layer LSTM Decoder（hidden size 512）
→ 8,000 個 Token Logits
→ Softmax / Decoding
```

#### LSTM Encoder 與 Transformer Encoder

| 面向 | HW5 LSTM Encoder | Transformer Encoder |
|---|---|---|
| 資訊傳遞 | `h_t` 依賴 `h_(t-1)`，依序讀取 | 每層用 Self-Attention 直接交換所有位置資訊 |
| 平行化 | 時間步之間有依賴，較難平行 | 同一層各位置可平行計算 |
| 順序資訊 | 遞迴結構自然帶入順序 | 需要 Positional Encoding |
| 長距離關係 | 必須經過多個時間步 | Attention 可直接連接遠距位置 |
| 核心狀態 | Hidden State + Cell State | 每個 Token 的 Context Representation |
| 目前狀態 | 已完成訓練 | 尚未在 HW5 實作 |

#### LSTM Attention 與 Transformer Cross-Attention

兩者都在做「Decoder 查詢 Encoder」，但內部形式不同：

| 面向 | HW5 LSTM Attention | Transformer Cross-Attention |
|---|---|---|
| Query | Decoder 當前 Hidden State | Decoder 子層表示經 Q projection |
| Key / Value | Encoder 每個 Hidden State | Encoder 表示經 K/V projections |
| Head 數 | Baseline 為單一 Attention 機制 | 通常是 Multi-Head |
| Decoder 執行 | 隨 LSTM 時間步逐步查詢 | 訓練時可對多個位置平行計算 |
| 共同目的 | 找出目前輸出最需要的輸入位置並形成 Context Vector | 相同 |

```text
score_i = similarity(decoder_state, encoder_output_i)
weights = softmax(masked_scores)
context = Σ weights_i × encoder_output_i
```

這裡遮掉的是 `<pad>`，避免注意無效補齊位置；Transformer Decoder 的 causal mask 則是避免偷看未來輸出，兩種 mask 的目的不同。

### 9.6 學習生命週期

```text
Forward：Embedding → Encoder → Attention → Decoder → Logits
→ Label-Smoothed Cross-Entropy
→ Backpropagation
→ Gradient Accumulation
→ Gradient Clipping
→ Adam + LR Scheduler
→ Parameter Update
```

目前重要設定：

| 技術 | 設定 | 所屬類別 | 目的 |
|---|---:|---|---|
| Teacher Forcing | 正確前一 Token 作為 Decoder input | Decoder 訓練策略 | 讓每個位置在正確前文下學習 |
| Label Smoothing | 0.1 | Loss Regularization | 降低過度自信，容許翻譯存在多種合理表達 |
| Adam | Optimizer | 參數更新 | 結合動量與自適應尺度 |
| Warmup | 4,000 updates | LR Scheduling | 訓練初期逐步提高 Learning Rate |
| Inverse Square Root | Warmup 後 | LR Scheduling | 隨更新次數逐步降低 Learning Rate |
| Gradient Clipping | norm 1.0 | 數值穩定 | 降低 RNN Gradient Explosion 風險 |
| FP16 | 啟用 | 計算精度／資源技巧 | 降低 VRAM 使用並提升運算效率 |

#### 普通 Cross-Entropy 與 Label-Smoothed Cross-Entropy

| 面向 | Cross-Entropy | Label-Smoothed Cross-Entropy |
|---|---|---|
| Target | 正確類別機率視為 1，其餘為 0 | 大部分機率給正確類別，少量分散給其他類別 |
| 模型傾向 | 可能變得非常確信 | 抑制過度自信 |
| 翻譯情境 | 假設唯一標準答案 | 較符合相同語意可能有多種表達 |

#### Mini-batch、Gradient Accumulation、Effective Batch

```text
max_tokens = 2,048
update_freq = 8
effective tokens/update ≈ 16,384
```

Gradient Accumulation 不等於把 16,384 tokens 同時放進 GPU；它是連續執行 8 個小 Batch 的 Forward/Backward，累積 Gradient 後才做一次 Optimizer Step，以適應 RTX 3050 的 4 GB VRAM。

### 9.7 推論與評估生命週期

| Checkpoint | 主要用途 |
|---|---|
| `checkpoint_last.pt` | 從最新訓練狀態續訓 |
| `checkpoint_best.pt` | 使用最佳 Validation Loss 模型推論 |
| `checkpoint40.pt` | 固定保存 Epoch 40 狀態 |

Checkpoint 除了 Weight，通常也保存 Epoch、Update、Optimizer、Scheduler 與最佳指標狀態。

目前解碼使用 `beam=5`：每一步最多保留五條累積 log-probability 較高的候選前綴；`nbest=1` 表示最後仍只輸出一條翻譯。Greedy 可視為 `beam=1`。

#### Validation Loss、BLEU、人工檢查

| 方法 | 衡量什麼 | 優點 | 限制 |
|---|---|---|---|
| Validation Loss | 正確下一 Token 的模型機率 | 適合選 Checkpoint，直接對應訓練目標 | 不等於整句翻譯品質 |
| BLEU | Hypothesis 與 Reference 的 n-gram 重疊 | 可量化整體翻譯結果 | 無法完整判斷語意與流暢度 |
| 人工抽查 | Source／Reference／Hypothesis 的實際品質 | 能發現語意錯誤、漏譯與不自然句子 | 主觀且無法大量執行 |

中文評估必須使用 SacreBLEU `tokenize='zh'`；不同 Tokenization 的 BLEU 不應直接比較。

### 9.8 已驗證實驗結果

| Epoch | Validation Loss | 中文 BLEU |
|---:|---:|---:|
| 10 | 5.689 | 18.43 |
| 30 | 5.408 | 20.43 |
| 40 | 5.370 | 20.64 |

判讀：

- Epoch 10 → 30 有明顯改善。
- Epoch 30 → 40 只增加 0.21 BLEU，邊際收益開始下降。
- 尚不能只憑這三點宣稱 Overfitting，但繼續訓練的成本效益已降低。
- 可報告的是 **local validation BLEU 20.64**；不可稱為 official test BLEU，因為官方 JudgeBoi 已失效，公開 `test.zh` 只是佔位符。

### 9.9 已完成、已接觸、尚未實作

| 狀態 | 內容 |
|---|---|
| 已完成實驗 | 資料前處理、Docker/CUDA、LSTM Seq2Seq + Attention、40 epochs、Beam Search、中文 Validation BLEU |
| 已在實作中使用 | Embedding、Autoregressive Decoder、Teacher Forcing、Label Smoothing、Adam、Warmup、FP16、Gradient Accumulation |
| 已學理論但未在 HW5 實作 | Transformer Encoder／Decoder、Self-Attention、Masked Multi-Head Attention、Positional Encoding、Residual、LayerNorm、Transformer FFN |
| 尚未實作的改善法 | NAT、Copy Mechanism、Guided／Monotonic Attention、Scheduled Sampling、Transformer baseline |

因此目前最精確的進度說法是：

> 已完成 HW5 英中翻譯的 LSTM Seq2Seq + Attention baseline，並已用 Transformer 課堂架構進行對照；完整 Transformer baseline 仍是下一個實驗。

### 9.10 LSTM 與 Transformer 要如何公平比較

不能只換架構就直接比較一次結果，至少要固定：

- 同一 Train／Validation split。
- 同一 SentencePiece model 與 Vocabulary。
- 同一中文 SacreBLEU `tokenize='zh'`。
- 相近參數量或清楚報告模型大小。
- 相同 Beam Search 設定。
- 同時報告 Training Time、VRAM、Validation Loss、BLEU 與人工樣本。

Transformer 常具備較好的長距離建模與訓練平行化能力，但在 4 GB VRAM 下可能被迫縮小 Batch、Layer、Head 或 FFN；所以「Transformer 一定較好」必須由同條件實驗證明。

---

## 10. 從資料到結果：四條完整實例

### 10.1 一般多類別分類

```text
Features x
  ↓
Fully Connected Layers
  ↓
ReLU（hidden activation）
  ↓
Logits
  ↓
Softmax（類別互斥的輸出表示）
  ↓
Categorical Cross-Entropy（衡量錯誤）
  ↓
Backpropagation
  ↓
Adam / SGD（更新參數）
```

### 10.2 影像分類

```text
Image + Data Augmentation
  ↓
Convolution → BatchNorm → ReLU
  ↓
Pooling / Strided Convolution
  ↓
多層 CNN Feature Extractor
  ↓
FC Classification Head
  ↓
Logits → Softmax + Cross-Entropy
  ↓
Backward → Optimizer Update
```

### 10.3 序列標記

```text
Sequence Embeddings + Positional Encoding
  ↓
Q/K/V Projections
  ↓
Multi-head Self-Attention
  ↓
每個位置取得 Context-aware Representation
  ↓
每個位置的 Classifier
  ↓
每個位置的 Logits → Softmax + Cross-Entropy
  ↓
Backward → Optimizer Update
```

### 10.4 Transformer Seq2Seq 生成

```text
Input Tokens + Positional Encoding
  ↓
Encoder Self-Attention + FFN（重複 N Blocks）
  ↓
Encoder Context Representations
  ↓
Decoder Masked Self-Attention
  ↓
Cross-Attention 讀取 Encoder 資訊
  ↓
Decoder FFN → Vocabulary Logits → Softmax
  ↓
Teacher Forcing 計算每個位置的 Cross-Entropy
  ↓
Backward → Optimizer Update

推論時：<BOS> → Greedy / Beam Search 逐步生成 → <EOS>
```

---

## 11. 最容易混淆的對應關係

| 容易混淆的組合 | 正確區分 |
|---|---|
| ReLU vs Sigmoid | 同為 Activation；但 Sigmoid 也常作為二元輸出轉換 |
| Sigmoid vs Softmax | Sigmoid 各輸出獨立；Softmax 讓互斥類別共同競爭、總和為 1 |
| Softmax vs Cross-Entropy | Softmax 轉換 logits；Cross-Entropy 衡量預測與標籤的差異 |
| Loss vs Optimizer | Loss 定義「錯多少」；Optimizer 決定「參數怎麼改」 |
| Backprop vs Gradient Descent | Backprop 算 gradient；Gradient Descent 使用 gradient 更新參數 |
| Learning Rate vs Optimizer | LR 是步幅；Optimizer 是使用 gradient 與歷史資訊的更新規則 |
| Batch Size vs Epoch | Batch Size 決定一次看幾筆；Epoch 是全部資料看過一次 |
| BatchNorm vs Batch Size | BatchNorm 是正規化元件；Batch Size 是資料分組超參數，但會影響 BN 統計品質 |
| Dropout vs BatchNorm | Dropout 主要抑制 overfitting；BN 主要改善特徵尺度與訓練穩定性 |
| CNN vs Convolution | CNN 是架構；Convolution 是其中的運算／Layer |
| CNN Filter vs Feature Map | Filter 是可學參數；Feature Map 是 Filter 掃描輸入後的輸出 |
| Attention Score vs Attention Weight | Score 是 QK 相似度；Weight 是經 Softmax 後的數值 |
| Q/K vs V | Q/K 決定關注誰；V 提供最後被匯總的內容 |
| Self-Attention vs Transformer | Self-Attention 是關係建模元件；Transformer 是包含 Attention、FFN、Residual、LayerNorm 與 Encoder／Decoder 結構的完整架構 |
| Encoder Self-Attention vs Decoder Masked Self-Attention | Encoder 可看完整輸入；Decoder 為避免答案洩漏，只能看目前與過去位置 |
| Self-Attention vs Cross-Attention | Self-Attention 的 Q/K/V 來自同一序列；Cross-Attention 的 Q 來自 Decoder，K/V 來自 Encoder |
| Encoder vs Decoder | Encoder 將整段輸入編碼成上下文表示；Decoder 根據已有輸出與 Encoder 表示逐步生成 |
| Residual Connection vs LayerNorm | Residual 提供資訊與梯度捷徑；LayerNorm 調整單筆資料內 feature 的尺度 |
| Teacher Forcing vs Beam Search | Teacher Forcing 是訓練時提供正確前文；Beam Search 是推論時搜尋輸出序列 |
| Greedy vs Beam Search | Greedy 每步只留一個最佳候選；Beam Search 同時保留多個累積分數較高的候選 |
| AT vs NAT | AT 逐 token 生成、依賴性強但慢；NAT 平行生成較快，但輸出協調與品質較困難 |
| Seq2Seq vs Transformer | Seq2Seq 是輸入輸出皆為序列的任務框架；Transformer 是實作這類任務的一種模型架構 |
| HW5 主題 vs HW5 實作模型 | 主題涵蓋 Transformer；目前真正完成的 Baseline 是 LSTM Seq2Seq + Attention |
| LSTM Attention vs Self-Attention | LSTM Attention 是 Decoder 查詢 Encoder；Self-Attention 是同一序列內各位置互相查詢 |
| Padding Mask vs Causal Mask | Padding Mask 排除 `<pad>`；Causal Mask 避免 Decoder 看見未來輸出 |
| Token ID vs Embedding | Token ID 是離散索引；Embedding 是可學習的連續向量 |
| SentencePiece vs fairseq Binary | SentencePiece 負責 Subword 切分；fairseq Binary 負責高速儲存與定位 Token ID |
| Label Smoothing vs Softmax | Label Smoothing 改變訓練 Target 分布；Softmax 將 Logits 轉成類別機率 |
| Gradient Accumulation vs Large GPU Batch | 前者分多次累積 Gradient，後者一次把全部資料放入 GPU；有效 Batch 可相近但記憶體與 BN 行為未必相同 |
| Validation Loss vs BLEU | Loss 衡量 Token 預測目標；BLEU 衡量生成句與參考句的 n-gram 重疊 |
| Local Validation BLEU vs Official Test BLEU | 前者可用真實 Validation reference 計算；HW5 官方測試目前無法重現，不能混稱 |
| Parameter vs Hyperparameter | Parameter 由訓練學得；Hyperparameter 由人或搜尋流程設定 |

---

## 12. 技術定位卡：看到名詞時的固定回答法

遇到任何技術，依序回答：

```text
1. 所屬大類是什麼？
2. 位於生命週期哪裡？
3. 上游輸入是什麼？
4. 核心處理是什麼？
5. 下游輸出交給誰？
6. 同類替代方法有哪些？
7. 為什麼在這個情境選它？
8. 限制或常見錯誤是什麼？
```

### 範例：ReLU

- 所屬大類：Activation Function。
- 位置：Linear 或 Convolution 之後的 hidden representation。
- 輸入：`z = Wx + b`。
- 處理：`max(0,z)`。
- 輸出：非線性 feature，交給下一層。
- 同類：Sigmoid。
- 選擇理由：計算簡單，正區域梯度較容易傳遞。
- 限制：負區域梯度為 0。

### 範例：Adam

- 所屬大類：Optimizer。
- 位置：Backpropagation 算完 gradient 之後。
- 輸入：目前 gradient 與歷史統計。
- 處理：結合 Momentum 的方向資訊與 RMSProp 的自適應尺度。
- 輸出：新的模型參數。
- 同類：SGD、Momentum、Adagrad、RMSProp。
- 選擇理由：通常是容易開始的通用 optimizer。
- 限制：仍需選 learning rate，且不保證所有任務泛化最佳。

### 範例：Batch Normalization

- 所屬大類：Normalization／訓練穩定化。
- 位置：模型中間層，常放在 Linear/Convolution 與 Activation 附近。
- 輸入：一個 Batch 的中間 feature。
- 處理：用 Batch 統計量正規化，再透過可學習參數調整尺度與位移。
- 輸出：尺度較穩定的 feature。
- 同類：Layer、Instance、Group Normalization。
- 選擇理由：Batch 足夠大且中間 feature 分布需要穩定時。
- 限制：小 Batch 的統計較不可靠；Inference 必須使用訓練累積的統計量。

### 範例：Cross-Attention

- 所屬大類：Attention／Encoder-Decoder 資訊交換。
- 位置：Transformer Decoder 的 Masked Self-Attention 之後。
- 輸入：Decoder 表示作為 Query；Encoder 輸出作為 Key 與 Value。
- 處理：Decoder 依目前生成需求，選擇輸入序列中相關資訊。
- 輸出：融合輸入內容的 Decoder representation。
- 同類比較：Self-Attention 的 Q/K/V 來自同一序列。
- 選擇理由：輸出必須受另一段輸入序列控制時。
- 限制：依賴 Encoder 提供有用的表示，並增加額外計算。

---

## 13. 口頭複習題

### 第一層：分類定位

1. ReLU、Sigmoid、Softmax 各屬於什麼類別？為什麼不能永遠把三者視為完全相同角色？
2. Cross-Entropy 和 Adam 分別負責訓練閉環中的哪一步？
3. BatchNorm、Dropout、Data Augmentation 都是訓練技巧，但介入位置有何不同？
4. Convolution、CNN、Filter、Feature Map 的層級關係是什麼？
5. Q、K、V 是 Parameter、Hyperparameter，還是中間表示？對應的投影矩陣又是哪一種？
6. Self-Attention、Transformer Block、Transformer 三者的層級關係是什麼？
7. Teacher Forcing、Scheduled Sampling、Beam Search 分別位於訓練或推論的哪一階段？

### 第二層：前後串接

1. 從一張影像進入 CNN 到參數更新，依序經過哪些步驟？
2. Softmax 前後的資料分別叫什麼？Cross-Entropy 接在哪裡？
3. Backpropagation 與 Optimizer 誰先誰後？各自產生什麼？
4. Self-Attention 中，Attention Score 如何一路變成輸出向量？
5. 為什麼純 Self-Attention 還需要 Positional Encoding？
6. Transformer Encoder Block 中，Attention、Residual、LayerNorm、FFN 如何依序串接？
7. Decoder 如何把 Masked Self-Attention、Cross-Attention 與 Softmax 串成一次生成步驟？

### 第三層：選擇與診斷

1. Training Loss 很高時，如何區分 Model Bias 與 Optimization Issue？
2. Training Loss 低但 Validation Loss 高，應優先從哪一類技術處理？
3. 為什麼多標籤分類通常用 Sigmoid 而不是 Softmax？
4. Loss 不下降是否等於卡在 Local Minimum？還有哪些可能？
5. Small Batch 和 Large Batch 如何同時影響速度、Gradient Noise 與泛化？
6. CNN 和 Self-Attention 都能取得上下文，它們的 receptive field 如何不同？
7. 什麼情況選 Encoder-only、Decoder-only 或 Encoder-Decoder？
8. AT 與 NAT 在速度、依賴關係和輸出品質上如何取捨？
9. Greedy Decoding 與 Beam Search 為何都不保證找到真正的最佳句子？
10. Teacher Forcing 為何會造成 Exposure Bias？Scheduled Sampling 想改善什麼？
11. 為什麼目前 HW5 不能說「已實作 Transformer」？LSTM baseline 與 Transformer 共用哪些 Seq2Seq 骨架？
12. SentencePiece、fairseq-preprocess、fairseq-train 分別位於哪一個生命週期？
13. Label Smoothing 與一般 One-hot Target 有何差異？為何可能適合翻譯？
14. `max_tokens=2048`、`update_freq=8` 如何形成約 16,384 effective tokens？
15. 為什麼 Validation Loss 下降不保證 BLEU 等比例上升？

---

## 14. 一分鐘總結模板

> 機器學習先依任務決定輸出形式，再選擇可表示目標函數的模型。資料經過模型 Forward 得到 logits，依任務用 Sigmoid 或 Softmax 轉成適合的輸出，再以 MSE 或 Cross-Entropy 等 Loss 衡量錯誤。Backpropagation 負責計算 Gradient，SGD、Momentum 或 Adam 等 Optimizer 負責更新參數。訓練過程還要用 Learning Rate、Batch Size、Normalization 控制優化穩定性，並用 Data Augmentation、Regularization、Dropout、Early Stopping 改善泛化。CNN 透過局部 Receptive Field 與 Parameter Sharing 處理影像；Self-Attention 透過 Q、K、V 動態建立序列各位置的關係。Transformer 再把 Multi-Head Attention、Feed-Forward、Residual Connection 與 LayerNorm 組成 Encoder／Decoder Blocks；Decoder 使用 Masked Self-Attention 防止偷看未來，並透過 Cross-Attention 讀取 Encoder 資訊，最後以 Autoregressive 或 Non-Autoregressive 方法生成輸出。

---

## 15. 本階段邊界

### 已納入本筆記

- Machine Learning：Model → Loss → Optimization。
- Linear Model、Sigmoid、ReLU、Batch、Epoch、Overfitting。
- Model Bias、Optimization Issue、Cross Validation、Mismatch。
- Critical Point、Batch Size、Momentum、Adaptive Learning Rate、LR Schedule。
- Feature / Batch Normalization 與其他 Normalization 類別。
- Classification 的 Softmax 與 Cross-Entropy 配對。
- CNN：Receptive Field、Parameter Sharing、Filter、Pooling、典型結構。
- Self-Attention：Q/K/V、Attention Matrix、Multi-head、Positional Encoding，以及與 CNN/RNN 的比較。
- Transformer：Seq2Seq、Encoder／Decoder、Residual Connection、Layer Normalization、Masked Self-Attention 與 Cross-Attention。
- Decoder 訓練與推論：AT／NAT、Teacher Forcing、Exposure Bias、Scheduled Sampling、Greedy Decoding 與 Beam Search。
- Seq2Seq 改善方法：Copy Mechanism、Guided Attention、Monotonic／Location-aware Attention 與 Training Noise。
- HW5 實作對照：英中資料生命週期、SentencePiece、fairseq Binary、Embedding、LSTM Encoder-Decoder Attention、Label Smoothing、Gradient Accumulation、FP16、Checkpoint 與 BLEU。
- HW5 已驗證結果：LSTM Baseline 40 epochs，local validation loss 5.370，中文 BLEU 20.64；官方 Test BLEU 不可重現。

### 下一階段再展開

- GAN / WGAN。
- Transformer 更進階的 Pre-LN／Post-LN 差異與實作細節。
- Efficient Attention 與長序列複雜度改善方法。

這些內容目前只作為知識地圖的下一站，不假設已完成學習。
