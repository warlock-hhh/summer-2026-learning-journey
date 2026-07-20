# Week 1–2｜2026/07/06–2026/07/19

## 本期摘要

這兩週的主要成果是完成 ML2021 HW1、HW2 的基準模型與多組改良實驗，並以 C++ 完成 5 題 LeetCode 和 3 題程式設計基礎練習。Raspberry Pi 目前只完成作業系統映像燒錄，尚未進行首次開機與遠端連線驗證。

## 已完成

### ML2021 HW1：COVID-19 Cases Prediction

- 任務：使用表格資料預測 `tested_positive`，評估指標為 RMSE。
- 建立 sklearn Ridge Regression baseline。
- 執行 PyTorch DNN、特徵選擇、正規化、learning rate、weight decay、hidden layer 與 batch size 實驗。
- 最佳結果為 Ridge Regression：Public RMSE `0.89396`、Private RMSE `0.92892`。
- DNN 經特徵選擇和正規化後明顯改善，但仍未超越 Ridge。

核心觀察：模型越複雜不代表結果越好。這份資料是小型表格資料，線性關係較強，Ridge 的歸納偏差反而更適合；DNN 容易受到樣本數、特徵尺度與過擬合影響。

### ML2021 HW2：TIMIT Framewise Phoneme Classification

- 任務：以連續 11 個語音 frame 組成 429 維輸入，分類成 39 個 phoneme 類別。
- 完成資料載入、訓練、驗證、checkpoint 與 Kaggle submission 流程。
- 依序測試 epoch、ReLU、weight decay、Dropout、BatchNorm、learning-rate scheduler 與加寬 MLP。
- Kaggle Public accuracy 從 `0.55417` 提升至 `0.73628`；Private accuracy 為 `0.73610`。
- 最佳設定使用 ReLU、BatchNorm、Dropout `0.3`、Adam、weight decay 與 `ReduceLROnPlateau`。

核心觀察：ReLU 改善梯度傳遞；Dropout 與 weight decay 抑制過擬合；BatchNorm 穩定中間特徵分布；scheduler 在 validation loss 停滯時降低 learning rate。

### LeetCode C++

| 題號 | 題目 | 核心方法 | 時間複雜度 | 空間複雜度 |
|---:|---|---|---:|---:|
| 28 | Find the Index of the First Occurrence in a String | 滑動起點逐字比對 | O(nm) | O(1) |
| 125 | Valid Palindrome | 左右雙指標與字元正規化 | O(n) | O(1) |
| 392 | Is Subsequence | 雙指標依序匹配 | O(n) | O(1) |
| 80 | Remove Duplicates from Sorted Array II | 快慢指標、原地覆寫 | O(n) | O(1) |
| 167 | Two Sum II | 利用排序性質收縮左右指標 | O(n) | O(1) |

這五題的主要學習主線是「雙指標」。重點不是背模板，而是辨認題目是否具有已排序、首尾比較、依序匹配或原地覆寫等結構。

### C++／程式設計助教準備

完成 3 題基礎程式：

1. 美元轉日圓：變數、輸入輸出與乘法運算。
2. 水力發電功率：常數、單位換算與公式實作。
3. 時間與溫度公式：整數／浮點數除法、三次方運算與運算優先順序。

### Raspberry Pi

- 已完成：將 Raspberry Pi OS 映像燒錄至儲存裝置。
- 尚未驗證：首次開機、網路、SSH、套件環境、GPIO、Camera 與 YOLO inference。

## 遇到的問題與處理方式

- Colab 的 `!gdown`、`!unzip` 不能直接放進一般 Python 腳本：改成 Windows／PowerShell 可執行的資料路徑與流程。
- 舊版程式使用已移除的 `np.int`：改為 `np.int64` 或內建 `int`。
- CPU 版 PyTorch 無法使用 CUDA：重新確認 PyTorch、CUDA 與顯示卡驅動版本是否匹配。
- DNN 不一定勝過線性模型：保留 Ridge baseline，透過 validation 與 Kaggle 指標比較，而非只看模型複雜度。
- 訓練產生大型模型與預測檔：以 `.gitignore` 排除 checkpoint、資料集、虛擬環境與 submission，GitHub 僅保存程式和實驗摘要。

## 尚未完成／下期工作

1. Raspberry Pi 完成首次開機、網路與 SSH 連線驗證，留下系統版本與指令紀錄。
2. 為 5 題 LeetCode 補齊可編譯解答、測試案例及錯題重寫紀錄。
3. ML2021 HW2 加入 Early Stopping，並固定 random seed 檢查結果穩定性。
4. 比較 Dropout `0.25`、`0.30`、`0.35`，避免只用單次 Kaggle 分數選模型。
5. 釐清 429 維 frame context 的時間結構，評估 1D CNN 或序列模型是否比 MLP 更合適。

## 教授 Meeting 簡短說法

這兩週先把 ML 的完整實驗流程跑通。HW1 的表格迴歸中，Ridge 比 DNN 更好，讓我確認模型複雜度必須配合資料型態；HW2 則從基礎 MLP 開始，逐步加入 ReLU、正則化、BatchNorm 和 scheduler，把 Kaggle accuracy 從 55.4% 提升到 73.6%。C++ 部分完成 5 題 LeetCode，主要練習雙指標；Raspberry Pi 目前只完成映像燒錄，下一步是開機及 SSH 驗證。
