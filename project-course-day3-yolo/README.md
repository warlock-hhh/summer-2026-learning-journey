# Project Course Day 3 — YOLO

## Current Progress

- Raspberry Pi：已完成 OS 映像燒錄，尚未完成首次開機與 SSH 驗證。
- Windows 備援環境：完成 YOLO 鳥類圖片批次推論。

## YOLO Bird Detection Minimal Project

使用預訓練 YOLO26n 權重對 `images/` 資料夾執行批次物件偵測，程式只統計 `bird` 類別，並輸出 class、confidence、bounding box `xyxy`、每張圖片與全部圖片的鳥數，以及畫框後的圖片。

### Environment

```text
Python 3.12.13
Ultralytics 8.4.120
CPU-only PyTorch
Confidence threshold 0.25
```

### Run

自行準備已授權的測試圖片及 YOLO26n 權重：

```text
project-course-day3-yolo/
├─ detect_bird.py
├─ requirements.txt
├─ images/          # not committed
├─ weights/         # not committed
│  └─ yolo26n.pt
└─ outputs/         # not committed
```

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\detect_bird.py
```

### Verified Result

六張測試圖片共偵測到 8 隻鳥：五張各 1 隻，一張成功偵測 3 隻。所有 bird confidence 介於 `0.907`～`0.952`。

這只能證明最小推論流程在目前六張清楚圖片上可執行，不能視為模型整體準確率。尚需測試無鳥圖片、小物件、遮擋、複雜背景、不同 confidence threshold，以及記錄 false positive／false negative。

## Technical Notes

- 本次執行的是 pretrained inference，不是模型訓練。
- Confidence 是單一 detection 的信心，不是整體 accuracy。
- Bounding box 使用 `(x1, y1, x2, y2)`，座標原點位於圖片左上角。
- 在程式中指定 `YOLO_CONFIG_DIR`，避免 Ultralytics 因使用者設定目錄權限失敗。
- 權重、輸入圖片與輸出圖片不放入 repository；圖片公開前必須確認來源與授權。

## Next Checkpoint

1. 完成 Raspberry Pi 首次開機、網路與 SSH 驗證。
2. 測試 confidence threshold `0.25`、`0.94`、`0.96`。
3. 加入無鳥、其他動物、小鳥、遮擋與複雜背景圖片。
4. 將每張圖片的偵測統計輸出成 CSV。
5. 學習 IoU、Precision、Recall、mAP50 與 mAP50-95。

## Four-Hour Classroom Kit

已完成 Raspberry Pi × OpenCV × YOLO 四小時課堂教材，正式題目為「圖片多物件偵測與分類統計系統」。教材以書籍第 11 章為主線，但更新成專案 `.venv`、VS Code 與 Ultralytics YOLO26n。

教材內容位於 [`classroom-kit/`](classroom-kit/)：

- 課前準備與五道驗收
- 教師逐分鐘講稿
- 學生操作手冊
- 投影片逐頁內容
- 故障排除與離線備案
- 學生成果單
- 三階段學生程式與教師完成版

課堂由 OpenCV 讀圖開始，再進入 YOLO inference、class／confidence／bounding box，最後使用 `Counter` 完成多物件分類統計。Raspberry Pi 實機尚未完成開機、SSH、環境安裝及整包 dry run，因此目前狀態是「教材完成、實機未驗證」。
