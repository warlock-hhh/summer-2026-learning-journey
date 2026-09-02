# HW5 研究日誌：英翻中 Seq2Seq 實驗

> 本文件只保留兩類內容：可重現的實驗流程，以及理解實驗所必需的核心技術。
> 若日後有問題，直接在相關段落加入 `[!QUESTION]`，回答也繼續補在同一份文件。

---

# 0. 最終摘要

| 項目 | 結果 |
|---|---|
| 任務 | 英文翻譯成繁體中文 |
| 訓練資料 | TED2020，390,041 組英中句對 |
| 驗證資料 | 3,939 組英中句對 |
| 執行環境 | Windows + WSL2 + Docker Desktop + RTX 3050 Laptop GPU 4 GB |
| 實驗 1 | fairseq 內建 LSTM，最佳 BLEU 20.64 |
| 實驗 2 | 老師式 GRU + Attention，最佳 BLEU 18.60 |
| 實驗 3 | 4-layer Transformer，最佳 BLEU 23.59 |
| 最終模型 | Transformer Epoch 38 |
| 最佳 checkpoint | `checkpoints/transformer_medium/checkpoint_best.pt` |
| 評估限制 | JudgeBoi 已失效，只能報告 local validation BLEU |
| 結案決策 | 停止追加 epoch，不做 Back-translation，轉入原理與錯誤分析 |

## Meeting 報告用一句話

> 我先以老師式 GRU Seq2Seq 建立 Simple Baseline，再依作業提示把 Encoder／Decoder 改成 4-layer Transformer；在相同本機 validation set 上，BLEU 從 18.60 提升到 23.59，增加 4.99，接近課程 Medium Baseline 水準。

---

# 1. 作業生命週期

整份作業可縮成一條主線：

```text
原始英中句子
→ 清理與切分資料
→ SentencePiece subword
→ fairseq binary dataset
→ 建立 Seq2Seq 模型
→ 訓練與 validation
→ Beam Search 產生翻譯
→ BLEU + 人工案例評估
→ 保存最佳 checkpoint
```

老師講義的三階段 Workflow 與本專案對應如下：

| 老師 Workflow | 本專案實作 |
|---|---|
| Preprocessing | `preprocess_hw5.py`，清理、切分、SentencePiece、binary |
| Training | GRU、LSTM、Transformer 三條路線 |
| Testing | 使用 validation reference 計算中文 BLEU，並抽查翻譯 |

不能使用公開 `test.zh` 算 BLEU，因為它只是句點佔位符，真正答案原本由 JudgeBoi 保管。

---

# 2. 資料生命週期

## 2.1 平行語料

一筆資料必須成對：

```text
English : Cats are so cute.
Chinese : 貓咪真可愛。
```

模型從大量成對句子學習英文序列與中文序列的對應。切分結果：

```text
train : 390,041 pairs，用來更新模型參數
valid :   3,939 pairs，不更新參數，用來選模型
test  :   4,000 English inputs，中文答案不公開
```

## 2.2 清理與切分

`preprocess_hw5.py` 的責任：

1. 正規化文字與標點。
2. 移除空句、過長或長度比例異常的句對。
3. 將資料固定切成 train／valid。
4. 訓練 SentencePiece。
5. 將文字轉為 subword token。
6. 呼叫 fairseq-preprocess 產生 binary dataset。

驗證集必須從訓練資料中獨立保留，否則拿模型已經背過的句子評估，BLEU 會過度樂觀。

## 2.3 SentencePiece 與 Embedding 是不同階段

SentencePiece 處理「文字如何切開」：

```text
transportation
→ ▁trans port ation
→ token IDs
```

優點：

- vocabulary 固定為 8,000，不必為每個完整單字建立一格。
- 未見過的詞仍可拆成已知片段。
- 字首、字尾與專有名詞片段可以重用。

Embedding 處理「ID 如何變成模型可學習的向量」：

```text
Token ID 315
→ 查 Embedding matrix 第 315 列
→ 256 維向量
```

本作業使用 joined dictionary，英中共用 token 字典；但共用字典不代表所有 Embedding 權重必然共用。

## 2.4 fairseq binary

```text
.bin：連續保存 token IDs
.idx：記錄每句話在 .bin 的位置與長度
```

位置：`DATA/data-bin/ted2020/`

這讓每個 epoch 不必重新讀文字與切詞，可以快速組 batch。

---

# 3. 模型生命週期與核心原理

## 3.1 Seq2Seq 骨架

Seq2Seq 解決「輸入與輸出長度不同」的問題：

```text
英文 token sequence
→ Encoder：建立英文表示
→ Decoder：一次生成一個中文 token
→ 中文 token sequence
```

GRU、LSTM、Transformer 都能放進這個 Encoder–Decoder 骨架。差別不是任務改變，而是 Encoder／Decoder 內部如何傳遞資訊。

## 3.2 GRU 與 LSTM

兩者都是 RNN 的改良版本，會依序讀取 token：

```text
h_t = RNN(x_t, h_{t-1})
```

| 項目 | GRU | LSTM |
|---|---|---|
| 狀態 | hidden state | hidden state + cell state |
| 主要 gates | reset、update | forget、input、output |
| 結構 | 較精簡 | 記憶控制較細 |
| 一定較好？ | 否 | 否，必須實驗比較 |

### LSTM 的核心更新

```text
f_t = σ(W_f[x_t,h_{t-1}] + b_f)          # forget gate
i_t = σ(W_i[x_t,h_{t-1}] + b_i)          # input gate
o_t = σ(W_o[x_t,h_{t-1}] + b_o)          # output gate
g_t = tanh(W_g[x_t,h_{t-1}] + b_g)       # candidate memory
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t          # long-term cell state
h_t = o_t ⊙ tanh(c_t)                     # visible hidden state
```

`σ` 把數值壓到 0～1，像開關比例；`⊙` 是逐元素相乘。LSTM 把長期記憶 `c_t` 與當前輸出 `h_t` 分開管理。

### GRU 的核心更新

```text
z_t = σ(W_z[x_t,h_{t-1}])                 # update gate
r_t = σ(W_r[x_t,h_{t-1}])                 # reset gate
h~_t = tanh(W_h[x_t,r_t⊙h_{t-1}])         # candidate hidden
h_t = (1-z_t)⊙h_{t-1} + z_t⊙h~_t          # mix old/new state
```

GRU 沒有獨立 cell state，直接用 update gate 混合舊 hidden 與新 candidate，因此參數通常少於同 hidden size 的 LSTM。

老師 Simple Baseline 實際使用 GRU：

```text
英文 IDs
→ 256-d Embedding
→ Bidirectional GRU Encoder（每方向 hidden 512）
→ 1024-d Encoder outputs
→ Attention
→ Unidirectional GRU Decoder（hidden 1024）
→ 中文 logits
```

Encoder 可雙向讀完整英文；Decoder 必須單向，因為生成現在 token 時不能偷看未來輸出。

## 3.3 老師式 Attention

Decoder 生成每個中文 token 時，都重新查看所有英文位置：

```text
Decoder query
→ 和每個 Encoder output 計算 score
→ mask 掉 padding
→ Softmax 得到權重
→ 對 Encoder outputs 加權求和
→ context vector
```

公式：

```text
score_i   = query · key_i
weight_i  = softmax(score_i)
context   = Σ weight_i × value_i
```

這是 Decoder-to-Encoder Attention。它和 Transformer Cross-Attention 概念相近，但不是 Transformer Encoder 內部的 Self-Attention。

## 3.4 Transformer 改了什麼

RNN 必須依序處理：

```text
x1 → h1 → x2 → h2 → x3 → h3
```

Transformer 讓一句話中的各位置用 Self-Attention 直接交換資訊，因此訓練時更容易平行化，也較容易建立長距離關係。

### Query、Key、Value

可以用搜尋資料庫理解：

```text
Query：我現在想找什麼
Key：每個位置提供什麼索引
Value：該位置真正要取回的內容
```

Scaled Dot-Product Attention：

```text
Attention(Q,K,V) = softmax(QKᵀ / √d_k)V
```

除以 `√d_k` 是避免維度變大後分數過大，使 Softmax 過度集中、gradient 變差。

若 batch size 為 `B`、序列長度為 `T`、head 維度為 `d_k=64`：

```text
Q, K, V          : [B, heads=4, T, 64]
QKᵀ              : [B, 4, T, T]
attention weights: [B, 4, T, T]
weighted values  : [B, 4, T, 64]
concat heads     : [B, T, 256]
```

`[T,T]` attention matrix 是 Self-Attention 記憶體成本約為 `O(T²)` 的原因。

### Multi-Head Attention

本次 `d_model=256`、`heads=4`，所以每個 head 為 64 維。不同 heads 可同時學習不同關係，例如局部搭配、主詞動詞或長距離指涉。

### 三種 Attention

```text
Encoder Self-Attention
→ 英文每個位置查看整句英文

Decoder Masked Self-Attention
→ 中文位置只能查看自己和過去，不能偷看未來

Decoder Cross-Attention
→ 中文 Decoder 查看英文 Encoder outputs
```

### Positional Encoding

Self-Attention 本身不知道順序，所以必須加入位置資訊。本次使用 Sinusoidal Positional Encoding。

### Transformer block

每一層不是只有 Attention：

```text
LayerNorm
→ Multi-Head Attention
→ Residual connection
→ LayerNorm
→ FFN：256 → 1024 → 256
→ Residual connection
```

本次使用 4-layer Encoder + 4-layer Decoder、Pre-LayerNorm。

### 一層 Encoder 與 Decoder 的完整資料流

```text
Encoder layer:
x
→ LayerNorm
→ Multi-Head Self-Attention(x,x,x)
→ Dropout + Residual（加回 x）
→ LayerNorm
→ FFN：Linear(256,1024) → ReLU → Linear(1024,256)
→ Dropout + Residual

Decoder layer:
y
→ LayerNorm
→ Masked Self-Attention(y,y,y)
→ Residual
→ LayerNorm
→ Cross-Attention(Q=decoder, K/V=encoder outputs)
→ Residual
→ LayerNorm
→ FFN 256→1024→256
→ Residual
```

Residual 保留原訊息並改善深層 gradient 傳遞；LayerNorm 穩定每個 token 特徵的尺度；FFN 對每個位置獨立做非線性特徵轉換。

---

# 4. 學習生命週期

## 4.0 一個 batch 的 tensor 契約

fairseq iterator 交給模型的 batch：

```text
src_tokens         : [B, S]       英文 token IDs
src_lengths        : [B]          每句英文的有效長度
prev_output_tokens : [B, T]       右移後的中文 IDs
target             : [B, T]       要預測的正確中文 IDs
logits             : [B, T, V]    每個位置對 8,000 tokens 的未正規化分數
```

其中 `B` 是句子數、`S` 是該 batch 最長英文長度、`T` 是最長中文長度、`V=8000`。Padding 只為了形成矩陣，Loss 與 Attention 都必須 mask 掉 padding。

## 4.1 Teacher Forcing

訓練 Decoder 時，把正確中文序列右移一格作為輸入：

```text
target             : 我 喜歡 機器 學習 <eos>
prev_output_tokens : <eos> 我 喜歡 機器 學習
```

模型學的是「已知正確前文時，下一個 token 是什麼」。推論時沒有答案，只能把自己的預測送回下一步，這個落差稱為 exposure bias。

## 4.2 Forward、Loss、Backward、Update

一個 optimizer update：

```text
Forward：輸入 → Encoder → Decoder → logits
Loss：比較 logits 與 target
Backward：計算每個參數的 gradient
Gradient clipping：限制 norm，避免梯度爆炸
Optimizer step：AdamW 更新參數
Zero grad：清除已用 gradient
```

Backpropagation 透過鏈式法則把錯誤從 Decoder 傳回 Cross-Attention、Encoder 與 Embedding，因此整個模型一起學習。

## 4.3 Label smoothing

設定 `0.1`，不要求模型把正確 token 當成 100% 唯一答案。翻譯常有多種合理措辭，降低過度自信通常有助於泛化。

普通 cross entropy 對正確類別 `y`：

```text
L_NLL = -log p(y|x)
```

Label smoothing 把一小部分權重分給所有類別：

```text
L = (1-ε)L_NLL + (ε/V)Σ_j[-log p(j|x)]
ε = 0.1, V = 8000
```

`<pad>` 位置不計入 Loss。程式先對 logits 做 `log_softmax`，再 gather 正確 token 的 log probability。

## 4.4 Learning-rate scheduler

```text
前 4,000 updates：warmup，learning rate 線性增加
之後：依 update 次數的 inverse square root 下降
```

Transformer 訓練初期對 learning rate 敏感，warmup 可降低不穩定。

## 4.5 RTX 3050 記憶體策略

Transformer 正式設定：

```text
max_tokens       = 1,536
accum_steps      = 8
effective tokens ≈ 12,288 / optimizer update
precision        = FP16
clip_norm        = 1.0
```

每次只放較小 batch 進 4 GB GPU，累積 8 次 gradient 再更新，模擬較大有效 batch。

Gradient accumulation 的重點是：

```text
連續 8 個 micro-batches：只 backward，先不 optimizer.step()
→ gradient 相加
→ 除以累積的有效 token 數
→ clip_grad_norm_(1.0)
→ 做一次 AdamW update
```

FP16 減少 activation／gradient 記憶體，但小數值可能 underflow，所以 `GradScaler` 先放大 Loss，backward 後再 unscale gradient。

## 4.6 Smoke test

Smoke test 不追求模型變好，只用一個真實 batch 檢查：

```text
資料可讀
→ GPU 可用
→ Forward 成功
→ Loss 合理
→ Backward 成功
→ 記憶體沒有 OOM
```

Transformer 第一次 smoke test 發現初始 loss 221.57，遠高於隨機預測 8,000 tokens 的 `ln(8000)≈8.99`。原因是 Embedding 初始化尺度過大；改成 `std=d_model^-0.5` 後 loss 變成 9.73，才開始正式訓練。這證明 smoke test 能避免數小時後才發現設定錯誤。

---

# 5. 實驗日誌

## 5.0 技術與程式碼對應

| 技術 | 實際位置 |
|---|---|
| 雙向 GRU Encoder | `hw05_teacher_baseline.py` → `RNNEncoder` |
| Dot-product Attention | `hw05_teacher_baseline.py` → `AttentionLayer` |
| 單向 GRU Decoder | `hw05_teacher_baseline.py` → `RNNDecoder` |
| Encoder–Decoder 組合 | `hw05_teacher_baseline.py` → `Seq2Seq` |
| Label smoothing | `LabelSmoothedCrossEntropy` |
| AdamW + Noam schedule | `NoamOptimizer` |
| Forward／Backward／gradient accumulation | `train_one_epoch` |
| Validation／BLEU／Beam Search | `validate` + fairseq generator |
| Transformer Encoder／Decoder | fairseq `TransformerEncoder`、`TransformerDecoder`，由本地程式明確組裝 |
| Transformer 超參數與組裝 | `hw05_transformer_medium.py` → `transformer_config`、`build_transformer_model` |

## 5.1 實驗 A：fairseq 內建 LSTM baseline

### 目的

先確認 Docker、資料、GPU、checkpoint 與 BLEU 的完整流程可以運作。

### 執行方式

```powershell
.\train_baseline.ps1 -MaxEpoch 40
```

這條路線呼叫 `fairseq-train --arch lstm`；老師的 `hw05_zh.py` 沒有被執行，模型核心由 fairseq 套件提供。

### 結果

| Epoch | Validation loss | BLEU |
|---:|---:|---:|
| 10 | 5.689 | 18.43 |
| 30 | 5.408 | 20.43 |
| 40 | 5.370 | 20.64 |

### 判讀

Epoch 10→30 改善明顯；30→40 只增加 0.21 BLEU，單純增加 epochs 的邊際收益下降。

## 5.2 實驗 B：老師式 GRU + Attention

### 目的

把老師範例中的模型與訓練迴圈整理成一般 `.py`，讓 Encoder、Attention、Decoder、Loss 與 Backpropagation 可以直接閱讀。

### 程式

```text
hw05_teacher_baseline.py
train_teacher_baseline.ps1
```

### 執行方式

```powershell
.\train_teacher_baseline.ps1 -MaxEpoch 30
```

### 結果

```text
最佳 Epoch      : 28
Validation loss : 3.7750
BLEU            : 18.60
Epoch 30 BLEU   : 18.54
```

### 判讀

這是最接近老師 Simple Baseline 教學設計的版本。分數不是最高，但模型核心完全呈現在本地 Python，最適合用來學習 recurrent Seq2Seq。

## 5.3 實驗 C：Transformer Medium

### 假設

若把 recurrent Encoder／Decoder 改成 Transformer，Self-Attention 可直接建立全句關係，應能比 GRU／LSTM baseline 得到更高 BLEU。

### 一次修改的內容

```text
4-layer Transformer Encoder
4-layer Transformer Decoder
d_model = 256
heads = 4
FFN = 1024
Pre-LayerNorm + Residual
Positional Encoding
Attention dropout = 0.1
Activation dropout = 0.1
```

保留 SentencePiece、label smoothing、warmup scheduler、FP16、gradient accumulation、beam=5 與 best/last checkpoint。

### 執行方式

```powershell
.\train_transformer_medium.ps1 -MaxEpoch 40
```

### 結果

| Epoch | Validation loss | BLEU |
|---:|---:|---:|
| 36 | 3.5085 | 23.00 |
| 37 | 3.4927 | 23.32 |
| **38** | **3.4976** | **23.59** |
| 39 | 3.4918 | 23.50 |
| 40 | 3.4879 | 23.58 |

### 判讀

- 最佳 BLEU 在 Epoch 38，不是 loss 最低的 Epoch 40。
- Loss 衡量 token 機率；BLEU 衡量完整翻譯與 reference 的片段重疊，兩者不必同步。
- Epoch 38～40 已在 23.5 附近震盪，繼續增加 epochs 的效益有限。
- 最終使用 `checkpoint_best.pt`，不使用 `checkpoint_last.pt`。

---

# 6. 評估與結果分析

## 6.1 三模型比較

| 模型 | 最佳 Epoch | Local validation BLEU | 相對 GRU |
|---|---:|---:|---:|
| 老師式 GRU + Attention | 28 | 18.60 | - |
| fairseq LSTM | 40 | 20.64 | +2.04 |
| **4-layer Transformer** | **38** | **23.59** | **+4.99** |

這個比較證明「整套 Transformer Medium 設定」優於兩套 recurrent baselines；不能只據此宣稱 LSTM 一定勝過 GRU，因為各版本的 hidden size、Attention 與訓練實作也不同。

## 6.2 BLEU 怎麼看

BLEU 不是正確率。它計算 1～4 gram precision，再加入 brevity penalty，避免模型只輸出很短句子。

課程門檻：

| 等級 | Public | Private |
|---|---:|---:|
| Simple | 18.43 | 17.61 |
| Medium | 24.04 | 23.43 |
| Strong | 29.32 | 28.27 |

本機 Transformer BLEU 23.59 已接近 Medium，但 validation set 與當年 JudgeBoi hidden test 不同，所以只能說「接近 Medium 水準」，不能宣稱正式通過。

## 6.3 實際翻譯案例

| English | Hypothesis | Reference | 判讀 |
|---|---|---|---|
| `this is seaweed .` | 這是海草。 | 這是海草。 | 完全正確 |
| `that's not true .` | 這不是真的。 | 但情況不是這樣。 | 語意合理但字面不同，BLEU 可能低估 |
| `which really bothers me .` | 這真的讓我感到困擾。 | 我真的不喜歡這樣 | 自然且語意合理 |
| `he nodded .` | 他結婚了。 | 他點頭。 | 關鍵動詞嚴重錯譯 |

因此最終評估必須同時使用：

```text
Validation loss：模型機率是否穩定
BLEU：整個資料集的可比較分數
人工抽查：是否錯譯、漏翻、重複或語意偏離
```

## 6.4 Greedy 與 Beam Search

```text
Greedy：每一步只留最高機率 token，相當於 beam=1
Beam=5：每一步保留累積分數最高的五條前綴
```

Beam Search 能避免太早丟掉稍後可能變好的句子，但不保證找到全域最佳，也不保證 beam 越大 BLEU 越高。本作業最後 `nbest=1`，仍只輸出最高分的一句。

序列分數使用 token log probabilities 累加：

```text
score(y_1...y_T) = Σ_t log p(y_t | y_<t, x)
```

每一步把目前 5 條前綴分別擴展 vocabulary，再只留下累積分數最好的 5 條，直到產生 `<eos>`。

## 6.5 BLEU 的核心計算

BLEU 比較 hypothesis 與 reference 的 1～4 gram precision，並限制同一片段不能超過 reference 中的出現次數。幾何平均會懲罰任一階 n-gram 很差的模型：

```text
BLEU = BP × exp(Σ_{n=1..4} w_n log p_n)
```

若模型故意輸出很短句子來提高 precision，brevity penalty 會扣分：

```text
BP = 1                 if hypothesis length > reference length
BP = exp(1-r/c)        if c ≤ r
```

中文評估固定使用 SacreBLEU `tokenize='zh'`，否則不同 tokenization 會讓分數不可比較。

---

# 7. 環境、工具與檔案角色

## 7.1 Docker 在本作業做什麼

```text
Windows + NVIDIA Driver
└─ WSL2 Linux kernel
   └─ Docker Engine
      └─ ml2021-hw5 Image
         └─ Container：Python / PyTorch / CUDA runtime / fairseq
```

- Image 是安裝完成的環境模板，不是單一套件。
- Container 是 Image 啟動後的執行實例。
- Docker Desktop 必須開啟，因為 Windows 的 `docker` 指令只是 Client，真正執行 Container 的 Docker Engine 由 Desktop 管理。
- 專案透過 bind mount 出現在 Container 的 `/workspace`，所以 DATA 與 checkpoints 實際保存在 Windows，Container 刪除也不會消失。

## 7.2 fairseq 在三條路線中的角色

| 功能 | LSTM CLI | 老師式 GRU | Transformer Medium |
|---|---|---|---|
| Dictionary / binary / batch | fairseq | fairseq | fairseq |
| 模型核心 | fairseq 內建 | 本地 `.py` 寫 GRU/Attention | 本地 `.py` 組裝 Transformer blocks |
| Training loop | fairseq 內建 | 本地 `.py` | 沿用本地教學訓練迴圈 |
| Beam Search | fairseq | fairseq | fairseq |

Docker 解決「環境如何重現」；fairseq 解決「翻譯任務常用元件」；PyTorch 負責 tensor、自動微分與 GPU 計算。

## 7.3 `.ps1` 與 `.py`

```text
.py  ：真正的模型與訓練程式，用來閱讀與修改
.ps1 ：Windows 啟動器，負責開 Docker 並在 Container 中執行 .py
```

## 7.4 主要檔案

| 檔案／目錄 | 用途 |
|---|---|
| `preprocess_hw5.py` | 資料清理、切分與 SentencePiece |
| `hw05_teacher_baseline.py` | 老師式 GRU + Attention 教學版 |
| `train_teacher_baseline.ps1` | 啟動老師式 GRU 訓練 |
| `hw05_transformer_medium.py` | Transformer Medium 模型與實驗流程 |
| `train_transformer_medium.ps1` | 啟動 Transformer 訓練 |
| `train_baseline.ps1` | fairseq 內建 LSTM 啟動器 |
| `DATA/data-bin/ted2020/` | fairseq binary dataset |
| `checkpoints/teacher_gru_baseline/` | GRU checkpoints |
| `checkpoints/rnn_baseline/` | fairseq LSTM checkpoints |
| `checkpoints/transformer_medium/` | Transformer checkpoints 與翻譯 TSV |

---

# 8. 最終重現方式

## 8.1 前提

1. 啟動 Docker Desktop，底部顯示 Engine running。
2. 在 VS Code 開啟 `ml_hw5` 資料夾。
3. 在該資料夾的 PowerShell Terminal 執行指令。

## 8.2 Smoke test

```powershell
.\train_transformer_medium.ps1 -SmokeTest
```

## 8.3 從頭訓練 Transformer 40 epochs

```powershell
.\train_transformer_medium.ps1 -MaxEpoch 40 -NoResume
```

若不加 `-NoResume`，程式會從 `checkpoint_last.pt` 接續到指定的 MaxEpoch。

## 8.4 RTX 3050 OOM 備案

```powershell
.\train_transformer_medium.ps1 -MaxEpoch 40 -MaxTokens 1024 -AccumSteps 12
```

`1024 × 12 = 12,288`，有效 token 數大致維持不變。

## 8.5 最終可重現性摘要

```text
Task             : English → Traditional Chinese
Framework        : fairseq 1.0.0a0+9a1c497
PyTorch          : 1.10.0
CUDA runtime     : 11.3
GPU              : RTX 3050 Laptop GPU, 4 GB
Model            : 4-layer Transformer Encoder-Decoder
Vocabulary       : 8,000 joined dictionary
d_model          : 256
Attention heads  : 4
FFN dimension    : 1,024
Dropout          : 0.3
Label smoothing  : 0.1
Optimizer        : AdamW, betas=(0.9,0.98), eps=1e-9
Scheduler        : inverse square root, warmup 4,000 updates
Max tokens       : 1,536
Accumulation     : 8
Precision        : FP16
Beam             : 5
Trainable params : 11,469,824
Best epoch       : 38
Validation loss  : 3.4976
Validation BLEU  : 23.5896, tokenize='zh'
Best checkpoint  : checkpoints/transformer_medium/checkpoint_best.pt
Samples          : checkpoints/transformer_medium/validation_samples_epoch38.tsv
```

---

# 9. 結論與後續複習

## 9.1 最終結論

1. GRU、LSTM、Transformer 都是 Seq2Seq Encoder–Decoder 的不同內部實作。
2. 老師式 GRU 最適合理解 recurrent model、Attention 與手寫 training loop。
3. Transformer 透過 Self-Attention、Positional Encoding、Multi-Head Attention、FFN、Residual 與 LayerNorm，將 BLEU 從 18.60 提升至 23.59。
4. BLEU 23.59 接近課程 Medium 水準，但因 JudgeBoi test reference 不可得，只能報告 local validation。
5. Epoch 38～40 已進入平台期，繼續追加 epochs 或投入約 12 小時的 Back-translation，不符合目前時間效益。

## 9.2 建議複習順序

```text
資料如何變成 token IDs
→ Embedding
→ Seq2Seq Encoder–Decoder
→ GRU/LSTM hidden state
→ 老師式 Attention
→ Self-Attention Q/K/V
→ Transformer Encoder / Decoder
→ Teacher Forcing / Loss / Backpropagation
→ Beam Search / BLEU
→ Docker / fairseq 只是如何實作與重現
```

## 9.3 尚未實作但知道定位

- Back-translation：使用中翻英模型與中文單語資料產生合成平行語料，對應 Strong Baseline。
- Scheduled Sampling：訓練時偶爾讓 Decoder 使用自己的預測，緩解 exposure bias。
- Copy Mechanism：協助複製姓名、數字與專有名詞。
- Non-Autoregressive Transformer：平行生成輸出，但需處理長度與 multi-modality。

## 9.4 後續提問格式

```markdown
> [!QUESTION]
> 我不懂這一段的 ______，請用一個具體句子解釋。
```

---

## 最後一句話

```text
資料處理把英中句子變成 subword IDs；
Encoder 理解英文，Decoder 產生中文；
GRU/LSTM 依序傳遞記憶，Transformer 用 Attention 建立全句關係；
Loss 與 AdamW 讓模型學習，Beam Search 生成翻譯；
BLEU 與人工案例共同判斷品質；
本次完成從 Simple GRU 18.60 到近 Medium Transformer 23.59 的完整實驗。
```
