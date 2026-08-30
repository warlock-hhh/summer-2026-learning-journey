# Week 7–8｜2026/08/17–2026/08/30

## 本期摘要

這兩週完成 ML2021 HW5 英翻中 Seq2Seq 實驗，依序建立 fairseq LSTM、老師式 GRU + Attention 與 4-layer Transformer，最終 local validation BLEU 為 `23.59`。LeetCode 完成 6 題，其中 5 題為 Binary Tree、1 題為運算式解析。專題課完成 Raspberry Pi × OpenCV × YOLO 四小時教材包，但尚未在 Raspberry Pi 實機完成環境與全流程驗證。C++ TA 本期沒有新增進度。

## 已完成

### ML2021 HW5：English-to-Chinese Translation

- 完成 TED2020 平行語料清理、切分、SentencePiece 與 fairseq binary preprocessing。
- 建立 Docker／WSL2／GPU 可重現環境。
- 完成 recurrent Seq2Seq 與 Transformer 三條實驗路線。
- 使用 validation loss、BLEU 與人工翻譯案例共同評估。

| 模型 | 最佳 Epoch | Local Validation BLEU |
|---|---:|---:|
| 老師式 GRU + Attention | 28 | 18.60 |
| fairseq LSTM | 40 | 20.64 |
| 4-layer Transformer | **38** | **23.59** |

Transformer Epoch 38～40 已在 BLEU 23.5 附近平台化，因此停止追加 epoch，也沒有投入約 12 小時執行 Back-translation。JudgeBoi 已失效，這次只能報告本機 validation BLEU，不能宣稱正式通過原課程 hidden test。

### LeetCode

完成題目：`104`、`100`、`224`、`101`、`112`、`105`。

- DFS 深度：104 Maximum Depth of Binary Tree
- 同位置樹比較：100 Same Tree
- Stack／Expression Parsing：224 Basic Calculator
- 鏡像比較：101 Symmetric Tree
- Root-to-leaf 路徑：112 Path Sum
- Traversal 重建樹：105 Construct Binary Tree from Preorder and Inorder Traversal

核心學習是先明確定義遞迴函式的回傳意義與 base case。題目 105 則利用 preorder 決定 root、inorder 切分左右子樹，並用索引表維持 O(n)。

### YOLO 四小時課堂教材包

- 完成課前準備與五道驗收流程。
- 完成四小時教師逐分鐘講稿。
- 完成學生操作手冊與投影片逐頁內容。
- 完成故障排除、離線備案與學生成果單。
- 完成 OpenCV 讀圖、YOLO inference、物件計數 starter 與教師完成版。
- 教學主線涵蓋 classification／detection／segmentation、confidence、bounding box、threshold、IoU 與 NMS。

教材將舊版 virtualenvwrapper、Thonny、YOLOv3／Darknet 流程更新成專案 `.venv`、VS Code 與 Ultralytics YOLO26n。這是教材完成，不代表 Raspberry Pi 實機已驗證。

## 技術理解與問題處理

1. SentencePiece 負責將文字切成 subword IDs；Embedding 再將 ID 映射成模型可學習的向量。
2. GRU／LSTM 依序傳遞 hidden state；Transformer 用 self-attention 直接建立全句關係。
3. Decoder 訓練使用 teacher forcing，推論時改用自己的輸出，因此存在 exposure bias。
4. BLEU 比較 n-gram 重疊，不是正確率；語意合理但措辭不同也可能被低估。
5. Transformer smoke test 曾出現 loss `221.57`，修正 embedding 初始化尺度後降到 `9.73`，避免錯誤設定進入長時間訓練。
6. RTX 3050 4 GB 使用 FP16 與 gradient accumulation，兼顧記憶體限制與有效 batch 大小。
7. 教學現場不應依賴首次下載權重；圖片、權重、虛擬環境及備援專案應在課前準備完成。

## 尚未完成／下期工作

1. Raspberry Pi 完成首次開機、網路、SSH 與 VS Code／Python 環境驗證。
2. 在每台 Pi 執行教材的五道驗收與完整四小時 dry run。
3. 記錄 YOLO 在 Pi 上的推論時間、記憶體使用與可能的 CPU 效能瓶頸。
4. 為 6 題 LeetCode 補齊可編譯解答、邊界測試與隔天重寫紀錄。
5. 複習 HW5 的資料生命週期、teacher forcing、attention、beam search 與 BLEU。
6. C++ TA 本期無新增；若暑期計畫繼續，再決定是否銜接 CH5 迴圈。

## 教授 Meeting 簡短說法

這兩週完成 ML2021 HW5 英翻中。我先跑 recurrent Seq2Seq baseline，再以 4-layer Transformer 替換 Encoder／Decoder，在相同 validation set 上把 BLEU 從老師式 GRU 的 18.60 提升到 23.59。LeetCode 完成 5 題 Binary Tree 和 1 題運算式解析。專題課方面完成 Raspberry Pi × OpenCV × YOLO 四小時教材包，包含教師講稿、學生手冊、分段程式、驗收與故障備案；不過目前只有教材完成，Raspberry Pi 實機尚未跑過完整流程。
