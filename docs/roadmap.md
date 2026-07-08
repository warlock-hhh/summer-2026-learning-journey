# 2026 暑假研究生能力養成 Roadmap v3

> 期間：2026/07/06–2026/08/30，共 8 週  
> 時間：週一至週五每天約 5–6 小時，週末 1–2 小時輕量複習  
> 核心修正：專題課共五天，我只負責 Day 3 的 Raspberry Pi YOLO 教學；不再準備完整五天課程，也不再把 Dashboard 當作必要成果。

---

## 1. 最終任務結構

暑假工作分成兩大部分，並同步進行。

### Part A｜個人能力養成

1. 李宏毅《機器學習 2021》40 部影片。
2. LeetCode C++ 平日每日練習。
3. Linux／Raspberry Pi 基礎操作與除錯能力。
4. Git／GitHub 管理程式、備課教材與學習紀錄。

### Part B｜助教備課

#### B1. 程式設計（一）

- 語言：C++。
- 依老師正式課綱、講義及前任助教教材準備。
- 交付：教案、範例、練習、解答、測資、rubric、FAQ、試教。

#### B2. 專題課 Day 3

- 專題課共五天，我只負責第三天。
- 主題：YOLO 基本概念、預訓練模型套用、完成小成果。
- 交付：YOLO 概念講義、Pi 推論範例、學生實作單、解答、FAQ、備援方案、完整 dry run。

---

## 2. 五天專題課流程與責任邊界

| 天數 | 負責人 | 內容 | 我的責任 |
|---|---|---|---|
| Day 1 | 學長 | Linux、Raspberry Pi、GPIO／LED 基礎 | 必須理解，確保能承接環境 |
| Day 2 | 另一位學長 | App 應用與其專業內容 | 了解留下的環境與操作方式 |
| Day 3 | 我 | YOLO 概念、模型套用、小成果 | 完整備課與授課 |
| Day 4 | 其他領域 | 不同領域內容 | 不需準備 |
| Day 5 | 其他領域 | 不同領域內容 | 不需準備 |

### 需要和 Day 1、Day 2 學長確認

- Raspberry Pi 型號、RAM、數量與分組方式。
- Raspberry Pi OS 版本、32-bit 或 64-bit。
- Python 版本。
- Day 1 是否完成 SSH、Git、Python、venv。
- Day 2 安裝哪些套件，留下哪些 App 或專案檔。
- Day 3 實際課程時間。
- 教室是否能連外下載模型與套件。
- 是否有相機；相機數量是否足夠。
- 是否可以事先把模型、圖片與安裝檔放入 SD 卡。

Day 3 不應重新教完 Day 1 的所有內容，但必須準備「環境檢查清單」和「快速修復方法」。

---

## 3. Day 3 YOLO 課程的合理範圍

### 學生下課後應能做到

1. 說明 Classification 與 Object Detection 的差異。
2. 說明 YOLO 的用途。
3. 理解 class、bounding box、confidence。
4. 使用預訓練模型對單張圖片執行 inference。
5. 找到並解讀輸出結果。
6. 修改 confidence threshold 並比較結果。
7. 完成一個可展示的小成果。

### 不列入標準要求

- 從零訓練 YOLO。
- 在 Raspberry Pi 上訓練模型。
- 自己標註大型資料集。
- 完整推導 YOLO loss。
- 比較所有 YOLO 版本。
- 每組都完成即時相機推論。

### 標準小成果

```text
輸入圖片
→ 載入預訓練 YOLO 模型
→ 執行物件偵測
→ 顯示 class / confidence / bounding box
→ 儲存結果圖片
→ 比較不同 confidence threshold
```

### 進階成果

- 短影片物件偵測。
- 計算每張圖片推論時間。
- 比較不同輸入尺寸或輕量模型。
- 教師示範即時相機推論。

相機不是標準實作必要條件；圖片與短影片可事先準備，避免硬體不足。

---

## 4. YOLO 部署策略

### 基本原則

- 訓練不放在 Raspberry Pi 上。
- Pi 的角色是載入預訓練模型並執行 inference。
- 先完成單張圖片，再嘗試影片，最後才是相機。
- 優先使用輕量模型。
- 在正式課堂前量測推論時間、RAM 使用及安裝時間。

### 三層備援

#### A 方案｜標準

Raspberry Pi 執行預訓練模型的單張圖片或短影片推論。

#### B 方案｜Pi 過慢或相容性問題

改由一般電腦執行相同 YOLO 程式，學生仍完成模型載入、推論、threshold 與結果分析。

#### C 方案｜無法連網或安裝失敗

- 預先下載模型與測試圖片。
- 預先準備套件安裝檔或可用環境。
- 提供已產生的輸出，至少完成程式閱讀、參數修改與結果分析。

### 參考資料

- [Ultralytics YOLO Predict 官方文件](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics Raspberry Pi 部署指南](https://docs.ultralytics.com/guides/raspberry-pi/)
- [Raspberry Pi 官方 AI 軟體文件](https://www.raspberrypi.com/documentation/computers/ai.html)
- [Raspberry Pi Getting Started](https://www.raspberrypi.com/documentation/computers/getting-started.html)
- [Raspberry Pi Remote Access](https://www.raspberrypi.com/documentation/computers/remote-access.html)

官方 Raspberry Pi 即時 AI 文件可能假設 Pi 5 與 Hailo NPU；實驗室若沒有 NPU，不應照該效能預期設計課程。

---

## 5. 個人學習教材

### 5.1 李宏毅《機器學習 2021》

- [中文版 40 部播放清單](https://youtube.com/playlist?list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J)
- 總長約 24 小時 13 分。
- 每週 5 部，8 週完成。
- 每部完成 Watch → Recall → Connect → Explain。

深度分級：

- 深入理解：訓練最佳化、CNN、Network Compression、Object Detection 相關概念。
- 理解用途：Transformer、GAN、Autoencoder、XAI、Domain Adaptation。
- 建立地圖：RL、Lifelong Learning、Meta Learning。

### 5.2 LeetCode C++

- 平日每天一題。
- 一題最多 45–60 分鐘，難題可拆成兩天。
- 八週目標約 35–40 題。
- 錯題在 48 小時及 7 天後重寫。

每題留下：

- Test cases。
- Brute force。
- Optimized idea。
- Time／space complexity。
- 使用的 STL。
- 第一次錯因。

題型順序：

```text
Array / String
→ Hash
→ Two Pointers
→ Binary Search
→ Stack / Queue
→ Linked List
→ Tree 基礎
```

### 5.3 Linux／Raspberry Pi

主要資源：

- [The Linux Command Line](https://linuxcommand.org/tlcl.php)
- [MIT Missing Semester](https://missing.csail.mit.edu/)
- Raspberry Pi 官方文件

因為只負責 Day 3，Linux 學習重點縮為 YOLO 部署真正會使用的能力：

- Pi OS、SSH。
- Shell、path、filesystem。
- Permission、user。
- `apt`、`pip`、venv。
- Git clone。
- Process、PID、基本 log。
- Python 程式執行與錯誤閱讀。
- CPU、RAM、溫度與效能限制。
- GPIO／LED 基本概念，確保能理解 Day 1。

systemd、Dashboard、深入 Bash、kernel driver 不再是暑假必要交付，但可作長期延伸。

---

## 6. 每日 Time Table

### 週一至週五，約 5 小時 30 分至 5 小時 45 分

| 時間 | 時長 | 工作 | 完成標準 |
|---|---:|---|---|
| 09:00–10:00 | 60 分 | LeetCode C++ | 程式、測試、複雜度、錯因 |
| 10:15–11:45 | 90 分 | 程式設計（一）備課 | 教案、範例、練習、試教 |
| 13:30–15:00 | 90 分 | ML2021 | 一部影片 + recall／術語整理 |
| 15:15–16:30 | 75 分 | Linux／Pi／YOLO | 環境、推論實作、教案或故障排除 |
| 16:30–16:45 | 15 分 | Git + Daily Note | commit、結論、明日第一步 |

專題課負擔降低後，Linux／Pi 時段從原本 90 分鐘縮為 75 分鐘；節省時間可用於程式設計備課、YOLO 實測或補 ML 筆記。

### 每週節奏

- 週一：新概念與本週交付物。
- 週二：跟做最小範例。
- 週三：關閉教材從零重做。
- 週四：加入測試、錯誤案例與 fallback。
- 週五：第五部 ML、試教／Demo、週報。
- 週末：錯題重寫、影片複習、下週準備；至少半天休息。

---

## 7. 八週同步 Roadmap

## Week 1｜07/06–07/12

### 個人學習

- ML2021 第 1–5 部：ML／DL 基本概念、任務攻略、batch、momentum。
- LeetCode：Array／simulation。
- C++：編譯、`vector`、function、reference 基礎。

### 程式設計備課

- 對齊 syllabus 與老師教材。
- 編譯流程、`main()`、`cout`／`cin`。
- 編譯錯誤與執行錯誤最小案例。
- 第一次 10 分鐘試講。

### 專題課 Day 3 準備

- 安裝 Raspberry Pi OS、設定 SSH。
- 練習 Shell、path、檔案操作。
- 理解 GPIO／LED 基本控制流程，重現 Day 1 可能使用的範例。
- 整理 Pi 環境與硬體資訊。

### 驗收

- ML 1–5 一頁摘要。
- LeetCode 至少 3–5 題。
- C++ 第一堂課教案草稿。
- Pi SSH 成功；能執行基本指令與理解 LED 範例。

## Week 2｜07/13–07/19

### 個人學習

- ML2021 第 6–10 部：learning rate、loss、BN、CNN、self-attention。
- LeetCode：String／Array traversal。

### 程式設計備課

- 型別、變數、constant、operator、I/O。
- Integer division、type conversion、overflow。
- 六題分層練習與測資。

### 專題課 Day 3 準備

- Python、`pip`、venv、requirements。
- Git clone、專案目錄與相對路徑。
- 建立 YOLO 前的環境檢查清單。
- 和學長確認 Day 1、Day 2 的環境交接。

### 驗收

- ML 6–10 摘要。
- LeetCode 累計約 10 題。
- 型別／運算式教案。
- 能在 Pi 建立 venv、安裝小型測試套件並執行 Python。

## Week 3｜07/20–07/26

### 個人學習

- ML2021 第 11–15 部：self-attention、Transformer、GAN／WGAN。
- LeetCode：Hash，熟悉 `unordered_map`、`unordered_set`。

### 程式設計備課

- Expression、operator precedence、comparison、logical operator。
- `=` vs `==`、short-circuit。
- Trace table 與 FAQ。

### 專題課 Day 3 準備

- Classification vs Object Detection。
- YOLO 的 input／output。
- Class、bounding box、confidence、threshold。
- Training vs inference、pre-trained model。
- 完成自己的 YOLO 概念筆記。

### 驗收

- Transformer 概念圖。
- Hash 題解與複雜度比較。
- Expression 教案。
- 10–15 分鐘 YOLO 基本概念試講。

## Week 4｜07/27–08/02

### 個人學習

- ML2021 第 16–20 部：GAN 應用、SSL、BERT。
- LeetCode：Two Pointers／Sliding Window。
- 隨機抽 5 題不看答案重寫。

### 程式設計備課

- Selection：`if`、nested if、`switch`。
- Range check、`break`、default、非法輸入。
- 第一次 20 分鐘正式試教。

### 專題課 Day 3 準備

- 先在一般電腦完成預訓練 YOLO 單張圖片 inference。
- 找到輸出結果、class、confidence、bounding box。
- 修改 threshold。
- 準備 5–10 張合法且容易辨識的測試圖片。

### 七月 Milestone

- ML2021 20／40。
- LeetCode 約 20 題，至少 5 題重寫。
- 程式設計備課完成至 Selection。
- 在一般電腦完成 YOLO 圖片推論並能解釋結果。

## Week 5｜08/03–08/09

### 個人學習

- ML2021 第 21–25 部：GPT、Autoencoder、Adversarial Attack。
- LeetCode：Binary Search／Stack。

### 程式設計備課

- Loop：`while`、`do-while`、`for`。
- Counter、sentinel、off-by-one、infinite loop。
- Nested loop trace table。

### 專題課 Day 3 準備

- 把單張圖片 YOLO inference 移到 Raspberry Pi。
- 記錄安裝時間、模型大小、推論時間、CPU／RAM。
- 判斷 Pi 是否能在課堂時間內穩定完成。
- 若失敗，記錄 root cause 並啟動 PC 備援方案。

### 驗收

- Autoencoder／Attack 概念比較。
- Loop 教案與題庫。
- Pi YOLO 單張圖測試報告；明確標記可行或不可行。

## Week 6｜08/10–08/16

### 個人學習

- ML2021 第 26–30 部：XAI、Domain Adaptation、RL、Policy Gradient。
- LeetCode：Queue／Linked List。

### 程式設計備課

- Function：declaration、definition、parameter、return、scope。
- Pass-by-value、reference、helper function。

### 專題課 Day 3 準備

- 完成學生版的圖片小成果。
- 時間允許再做短影片推論。
- 比較兩個 threshold。
- 建立無相機、無網路、Pi 過慢三種 fallback。
- 整理常見錯誤：模型找不到、圖片路徑錯、套件缺失、venv 未啟用。

### 驗收

- XAI／Domain Adaptation 一頁筆記。
- Function 教案。
- YOLO 小成果 v1 + fallback 測試。

## Week 7｜08/17–08/23

### 個人學習

- ML2021 第 31–35 部：RL 後半、Lifelong Learning。
- LeetCode：Tree 基礎或補強弱項。

### 程式設計備課

- Array、2D array、C++ `string`、`vector`。
- Boundary、index、輸入資料處理。
- 第二次 30 分鐘試教。

### 專題課 Day 3 準備

- 完成 YOLO Day 3 講義。
- 完成教師示範程式與學生版本。
- 完成實作單、解答、FAQ。
- 設計基本任務與進階挑戰。
- 進行一次完整試教。

### 驗收

- RL／Lifelong Learning 領域圖。
- Array／String 教案。
- YOLO Day 3 教材 package v1。

## Week 8｜08/24–08/30

### 個人學習

- ML2021 第 36–40 部：Network Compression、Meta Learning、總結。
- LeetCode：錯題 48 小時／7 天重寫，不大量開新題。
- 特別整理 Network Compression 與 Raspberry Pi 的關係。

### 程式設計備課

- 整合所有教案、範例、練習、答案、rubric、FAQ。
- 從學生角度完整 dry run。

### 專題課 Day 3 準備

- 從乾淨 Pi 環境重新執行完整流程。
- 測試離線模型與測試圖片。
- 演練套件安裝失敗、路徑錯誤、Pi 過慢、記憶體不足、無相機。
- 完成最後一次試教並記錄實際耗時。

### 八月 Final Milestone

- ML2021 40／40 + 8 份週摘要。
- LeetCode 約 35–40 題 + 重寫紀錄。
- 完整程式設計（一）TA package。
- 完整 YOLO Day 3 package + 三層備援 + dry run。

---

## 8. Day 3 建議課程流程

以下以約 3 小時為例，實際依學長提供的課程時間調整。

| 階段 | 時間 | 內容 |
|---|---:|---|
| 概念 | 25–30 分 | Classification、Detection、YOLO、class、box、confidence |
| 環境檢查 | 15–20 分 | Python、venv、套件、模型、圖片路徑 |
| 教師示範 | 20 分 | 載入模型、單張圖片 inference、儲存結果 |
| 學生實作 | 50–60 分 | 執行指定圖片、找輸出、修改 threshold、換圖片 |
| 小挑戰 | 25–30 分 | 推論時間、物件數量、短影片或不同 threshold |
| 成果分享 | 15–20 分 | 輸入、輸出、參數、錯誤與解法 |

### 分級驗收

#### 基礎通過

- 模型成功載入。
- 單張圖片推論成功。
- 找到輸出圖片。
- 能指出 class、confidence、bounding box。

#### 標準通過

- 更換測試圖片。
- 修改 threshold。
- 比較結果並解釋誤判或漏偵測。

#### 進階通過

- 短影片推論。
- 記錄推論時間。
- 比較不同模型或輸入尺寸。

---

## 9. 備課交付標準

### 程式設計（一）每個單元

1. Learning outcomes。
2. 課前診斷題。
3. 概念說明。
4. Live coding 與故意犯錯。
5. 基礎／標準／挑戰練習。
6. 解答、測資、rubric。
7. FAQ。
8. Exit ticket。

### YOLO Day 3

1. YOLO 基本概念講義。
2. 環境檢查清單。
3. 教師版完整程式。
4. 學生版 starter code。
5. 測試圖片與短影片。
6. 預先下載的模型。
7. 學生實作單與解答。
8. 常見錯誤 FAQ。
9. Pi／PC／離線三層 fallback。
10. 試教與完整 dry run 紀錄。

---

## 10. Git 與筆記結構

```text
summer-2026/
├─ README.md
├─ leetcode-cpp/
├─ ml2021/
│  ├─ weekly-notes/
│  └─ terminology.md
├─ cpp-ta/
│  ├─ lessons/
│  ├─ examples/
│  ├─ exercises/
│  └─ faq.md
├─ project-course-day3-yolo/
│  ├─ concepts/
│  ├─ environment-check/
│  ├─ teacher-demo/
│  ├─ student-lab/
│  ├─ assets/
│  ├─ troubleshooting/
│  └─ fallback/
└─ reports/
```

每個 YOLO 實驗記錄：

```markdown
# Experiment ID
- Date / Git commit：
- Hardware / RAM / OS：
- Python / package version：
- Model：
- Input：
- Command / code：
- Inference time：
- CPU / RAM：
- Result：
- Error：
- Root cause：
- Fix：
- 是否適合課堂：Yes / No / Conditional
```

---

## 11. 每週教授報告模板

```markdown
# Week N Report

## 1. 本週結論
完成 ___；證據是 ___；目前最大問題是 ___。

## 2. 個人學習
- ML2021：第 __–__ 部
- 核心概念：
- LeetCode C++：新題 __；重寫 __

## 3. 程式設計備課
- 完成章節：
- 教案／題庫／試教：
- 學生可能遇到的問題：

## 4. YOLO Day 3 備課
- 本週完成：
- 測試硬體與環境：
- Demo／程式／文件：
- 效能或錯誤：
- 是否需要 fallback：

## 5. 下週三個交付物
1.
2.
3.

## 6. 希望教授或學長確認
- 課程時間／設備／環境／實作難度：
```

---

## 12. 時間不足時的優先級

### P0｜必須完成

1. ML2021 指定 40 部。
2. LeetCode C++ 每日習慣，但題數可彈性。
3. 程式設計（一）核心教案與試教。
4. YOLO 單張圖片 inference。
5. YOLO Day 3 學生實作單、FAQ、fallback、dry run。
6. Git、README、錯誤紀錄。

### P1｜時間足夠再完成

- YOLO 短影片推論。
- Pi 上的模型格式或效能最佳化。
- 推論時間比較。
- 更多 LeetCode 題目與自動測試。

### P2｜可延後

- 每組即時相機推論。
- 自建資料集與模型訓練。
- Dashboard。
- systemd 完整服務化。
- 深入 kernel／driver／device tree。
- FTL／PCIe／NVMe 深入規格。

---

## 13. 暑假結束 Checklist

### 個人能力

- [ ] 完成 ML2021 40 部與 8 份週摘要。
- [ ] 能說明 training、inference、loss、optimizer、overfitting。
- [ ] 能說明 Classification 與 Object Detection 差異。
- [ ] 完成約 35–40 題 LeetCode C++。
- [ ] 錯題有 48 小時／7 天重寫。
- [ ] 熟悉基本 STL 與複雜度分析。

### 程式設計（一）

- [ ] 完成環境、I/O、型別、expression、selection、loop、function、array 教案。
- [ ] 每章有範例、練習、解答、測資、rubric、FAQ。
- [ ] 完成至少兩次試教。
- [ ] 能引導學生閱讀 compiler error。

### Linux／Raspberry Pi

- [ ] 能建立 Pi OS、SSH 與 Python venv。
- [ ] 能操作 shell、path、permission、package、process、log。
- [ ] 能重現 Day 1 的 Pi／LED 基礎。
- [ ] 能判斷 Pi CPU／RAM 是否足以執行模型。

### YOLO Day 3

- [ ] 能解釋 class、bounding box、confidence、threshold。
- [ ] 一般電腦的單張圖片推論成功。
- [ ] Raspberry Pi 的單張圖片推論已實測。
- [ ] 有 Pi、PC、離線三層 fallback。
- [ ] 教師版與學生版程式分離。
- [ ] 有測試圖片、實作單、解答、FAQ。
- [ ] 完成乾淨環境 dry run。
- [ ] 完成至少一次完整試教並記錄時間。

### 研究與工程習慣

- [ ] 所有成果可由 Git commit 追蹤。
- [ ] 環境、版本、指令與測試方式都有記錄。
- [ ] 能區分已驗證結果、合理推測與尚待確認事項。
- [ ] 報告以程式、數據、錯誤與證據為主，不只報告觀看進度。

