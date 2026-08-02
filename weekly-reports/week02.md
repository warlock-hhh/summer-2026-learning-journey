# Week 3–4｜2026/07/20–2026/08/02

## 本期摘要

這兩週完成 ML2021 HW3 Food-11 圖片分類，從原始 CNN baseline 逐步測試資料增強、正則化、學習率調整、架構變化與半監督學習，最後以五層 CNN 搭配 dynamic self-labeling，將 Kaggle Public accuracy 從 `48.387%` 提升至 `77.897%`。C++ 部分完成 9 題 Hash Table 題型。Raspberry Pi 與程式設計助教準備本期沒有新增進度。

## 已完成

### ML2021 HW3：Food-11 Image Classification

- 使用 CNN 將圖片分類為 11 種食物類別。
- 資料包含 3,080 張 labeled training images、660 張 validation images 與 3,347 張 testing images。
- 建立 checkpoint、training curves、validation 與 Kaggle submission 流程。
- 依序測試 augmentation、Dropout、weight decay、scheduler、GAP、Residual、pseudo-label、CutMix、Ensemble 與 TTA。
- 最後重新建立一致的 reference pipeline，避免只為分數持續疊加技巧。

### 實驗結果

| 實驗 | Valid accuracy | Private | Public |
|---|---:|---:|---:|
| 原始 baseline | 47.88% | 45.965% | 48.387% |
| Augmentation + Dropout 0.1 + Scheduler | 56.21% | 56.066% | 55.794% |
| GAP + Pseudo label | 64.09% | 64.196% | 64.277% |
| Residual + Pseudo label | 66.06% | 65.391% | 64.695% |
| GAP + Residual + CutMix Ensemble | 68.94% | 68.798% | 68.100% |
| 五層 CNN + Dynamic self-labeling | **76.36%** | **76.748%** | **77.897%** |

### Dynamic self-labeling 流程

1. 先以 3,080 張 labeled images 訓練 teacher。
2. validation accuracy 超過 70% 後啟動 self-labeling。
3. 對 6,786 張 unlabeled images 推論 Softmax confidence。
4. 只採用 confidence ≥ `0.90` 的預測。
5. 將 pseudo-labeled images 暫時加入 training set。
6. 每 5 epochs 重新推論並更新 pseudo labels。

Pseudo-label 在 epoch 190 首次啟動，加入 1,783 張圖片；訓練後期約選入 4,287 張。完整訓練共 500 epochs、約 8 小時，最佳 checkpoint 位於 epoch 443。

### LeetCode：Hash Table

完成題目：`383`、`205`、`290`、`242`、`1`、`202`、`219`、`49`、`128`。

本期練習涵蓋：

- 頻率統計：383、242
- 雙向映射／雙射：205、290
- Complement lookup：1
- 循環偵測：202
- 最後出現索引：219
- 分組 signature：49
- Hash set 與序列起點：128

核心觀念：Hash table 的價值是用額外空間換取平均 O(1) 查找，但必須先定義正確的 key、value 與不變量；不能只是看到查找就直接套 `unordered_map`。

## 技術理解與反省

1. Dropout 並非越高越好；早期 CNN 使用 `0.3` 反而造成 underfitting，`0.1` 較平衡。
2. Scheduler 在 validation loss 停滯時降低 learning rate，使 Public accuracy 從 `51.911%` 提升至 `55.794%`。
3. 強 augmentation 只增加輸入變化，不會創造新的語意標籤；小資料集仍需要控制模型容量或利用 unlabeled data。
4. Pseudo-label 的關鍵是 teacher 品質與 confidence threshold，而不是把所有預測直接當真實標籤。
5. Validation、Private 與 Public accuracy 接近，表示目前 validation split 具有一定代表性。
6. Ensemble、TTA 與 CutMix 並非必然提升，若同時加入太多技巧，就難以判斷真正有效的變因。
7. Hash table 題型需要先辨識資料關係：計數、映射、索引、分組或集合，再選擇 `unordered_map`、`unordered_set` 或固定大小陣列。

## 尚未完成／下期工作

1. 為 9 題 LeetCode 補齊可編譯 C++ 解答、測試案例、複雜度與隔天重寫紀錄。
2. 複習 HW3 的 model、augmentation、training loop 與 dynamic pseudo-label，確保能不看日誌完整說明。
3. Raspberry Pi 完成首次開機、網路及 SSH 驗證；目前仍停留在映像燒錄階段。
4. 下一份 ML 作業先建立最小可執行 baseline，再一次只調整一個主要變因。

## 教授 Meeting 簡短說法

這兩週主要完成 ML2021 HW3 的 Food-11 圖片分類。我先從原始 CNN 開始，測試 augmentation、Dropout 和 scheduler，之後也嘗試 Residual、CutMix、Ensemble 與 pseudo-label。中途發現同時堆疊太多技巧會很難判斷真正的改善來源，因此最後重新建立一套一致的五層 CNN 流程，等 validation accuracy 超過 70% 後才啟動 dynamic self-labeling，每 5 epochs 更新高信心 pseudo labels。最後 Kaggle Public accuracy 從 48.4% 提升到 77.9%。C++ 部分完成 9 題 Hash Table 題型；Raspberry Pi 這兩週沒有新進度。
