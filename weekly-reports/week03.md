# Week 5–6｜2026/08/03–2026/08/16

## 本期摘要

這兩週完成 ML2021 HW4 語者分類實驗，從單層 Transformer baseline 逐步調整 Encoder layers、attention heads、segment length、訓練 epochs 與 pooling，最終 Kaggle Public／Private 分數達到 `0.91833`／`0.91000`。專題課部分建立 YOLO 鳥類偵測最小專案，在 CPU-only 環境完成六張圖片的批次推論，共偵測 8 隻鳥。LeetCode 完成 6 題 Linked List；C++ TA 與 Raspberry Pi 實機本期沒有新增進度。

## 已完成

### ML2021 HW4：Speaker Classification

- 使用 `(T, 40)` log-Mel spectrogram 進行 600 類語者分類。
- 建立固定 train-validation split、checkpoint、training curves、推論與 Kaggle submission 流程。
- 逐步測試 Encoder layers、attention heads、segment length、epochs 與 pooling。

| 實驗 | Valid Acc | Public | Private |
|---|---:|---:|---:|
| 單層、2 heads、segment 128 | 60.46% | 0.73690 | 0.72166 |
| 2 Encoder layers | 66.13% | 0.79904 | 0.80000 |
| 4 Attention heads | 69.20% | 0.83285 | 0.81000 |
| Segment 192 | 76.63% | 0.86023 | 0.84666 |
| Segment 256 | 81.41% | 0.86714 | 0.85777 |
| 30 epochs | 83.73% | 0.89238 | 0.89611 |
| Attention Pooling | **86.85%** | **0.91833** | **0.91000** |

最終模型採用 2 layers、4 heads、segment 256、30 epochs 與 Attention Pooling。相較課程 baseline，Private 從 `0.75888` 提升到 `0.91000`。

### YOLO 鳥類偵測最小專案

- 建立獨立虛擬環境並安裝 Ultralytics。
- 使用 pretrained YOLO26n 執行 inference，沒有重新訓練模型。
- 從單張圖片擴充為資料夾批次推論，只統計 `bird`。
- 輸出 confidence、`xyxy` bounding box、單張及總鳥數。
- 六張圖片共偵測 8 隻鳥，其中一張成功偵測 3 隻。
- 在 CPU-only PyTorch 環境完成驗證，可作為無 GPU 備援方案。

### LeetCode：Linked List

完成題目：`138`、`141`、`21`、`2`、`92`、`19`。

- 節點映射與複製：138 Copy List with Random Pointer
- Floyd 快慢指標：141 Linked List Cycle
- Dummy node 與合併：21 Merge Two Sorted Lists
- 串列逐位加法與 carry：2 Add Two Numbers
- 區間原地反轉：92 Reverse Linked List II
- 固定距離雙指標：19 Remove Nth Node From End of List

核心觀念是把「節點本身」與「節點之間的連結」分開思考。修改 `next` 前必須先保存後續節點；使用 dummy node 可以統一 head 可能被替換或刪除的邊界情況。

## 技術理解與問題處理

1. Encoder layer 同時包含 self-attention、residual、LayerNorm 與 FFN。
2. 4 heads 是把 80 維拆成四個 20 維子空間，不是把總維度變成四倍。
3. Segment 越長能提供更多聲紋上下文，但 attention 成本約隨 `T²` 增加。
4. Scheduler 的 LR 降到零只代表預定訓練預算結束，不能據此斷定模型收斂。
5. Attention Pooling 讓模型學習重要 frames，避免被靜音或普通 frames 稀釋。
6. YOLO confidence 是單一 detection 的信心，不代表整體 accuracy。
7. PowerShell 找不到 `python` 時，改用明確執行檔建立並呼叫 `.venv`。
8. Ultralytics 設定目錄權限失敗時，在 import 前設定專案內的 `YOLO_CONFIG_DIR`。
9. 自動下載權重受限時，事先準備官方權重並從本地路徑載入。

## 尚未完成／下期工作

1. Raspberry Pi 完成首次開機、網路、SSH 與 Python 環境驗證。
2. YOLO 加入無鳥、其他動物、小物件、遮擋及複雜背景圖片。
3. 比較 confidence threshold `0.25`、`0.94`、`0.96`，記錄漏檢與誤檢。
4. 將 YOLO 每張圖片的統計輸出為 CSV。
5. 學習 IoU、Precision、Recall、mAP50 與 mAP50-95。
6. HW4 補上 padding mask，避免短語音 padding 參與 attention 與 pooling。
7. 為 6 題 Linked List 補齊可編譯解答、測試案例及隔天重寫紀錄。

## 教授 Meeting 簡短說法

這兩週主要完成 ML2021 HW4 語者分類。我逐步增加 Encoder layer、attention heads 和語音 segment 長度，再延長訓練並加入 attention pooling，最終 Private score 從課程 baseline 的 0.75888 提升到 0.91000。專題課完成 CPU-only 的 YOLO 鳥類批次偵測；LeetCode 完成 6 題 Linked List。C++ TA 與 Raspberry Pi 實機本期沒有新增進度。
