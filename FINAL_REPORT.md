# 2026 暑期訓練計畫結案報告

## 一、計畫資訊

- 執行期間：2026/07/06–2026/09/07 當週
- 結案文件補登日期：2026/09/20
- 執行身分：淡江大學電機工程學系碩士生
- 計畫定位：研究所入學前的程式設計、機器學習、嵌入式與教學準備
- 結案狀態：暑期階段結束；本 repository 轉為唯讀性質的學習作品集與歷程保存

## 二、結案摘要

本計畫以「實際執行、理解、留下紀錄、能夠重現」為主線。暑期內完成 ML2021 HW1～HW5 五個機器學習專案、26 題 LeetCode、C++ 程式設計 CH1～CH4 期中前備課，以及 Windows 上的 YOLO 圖片推論、Raspberry Pi × OpenCV × YOLO 四小時課堂教材包、GitHub 教材發布與實際授課。

8/30 之後至 9/7 當週沒有新增 ML、LeetCode 或 C++ TA 進度，後段工作集中在 Raspberry Pi 專題課，並完成實際授課。暑期計畫於該週實際結束，只是到 9/20 才補做結案文件。現有 repository 沒有完整保留每台 Raspberry Pi 的環境檢查、程式輸出與效能數據，因此本報告只確認「課程已完成」，不額外推定每台設備都通過相同的工程驗收。LeetCode 也沒有為每一題補齊本機可編譯程式、完整測試與隔天重寫紀錄。

## 三、目標達成度

| 原始目標 | 結案結果 | 判定 |
|---|---|---|
| 完成 ML2021 40 部影片並建立摘要 | 完成 HW1～HW5 的程式、實驗、研究日誌與成果整理；沒有足夠紀錄證明 40 部影片全數完成 | 部分達成 |
| 使用 C++ 持續練習 LeetCode | 完成 26 題並整理題型與核心方法；未替每題保存完整本機程式與重寫紀錄 | 部分達成 |
| 準備程式設計（一）助教教材 | 完成 CH1～CH4 期中前教學攻略、常見錯誤、練習與驗證流程 | 達成暑期範圍 |
| 準備 Raspberry Pi YOLO 專題課教材與備援方案 | 完成 Windows 推論、教師講稿、學生手冊、分段程式、故障排除、離線備案、公開 GitHub 教材包與實際授課 | 達成 |
| 建立可重現、可說明、可除錯的工程習慣 | 建立 README、requirements、Docker、研究日誌、實驗表、Git 安全排除、Secret Scanning 與 pre-commit 掃描 | 達成 |

## 四、主要成果

### 1. Machine Learning：ML2021 HW1～HW5

| Project | Problem | Final evidence | Main learning |
|---|---|---:|---|
| [HW1](ml2021/hw01-covid-regression/README.md) | COVID-19 tabular regression | Private RMSE `0.92892` | 簡單模型可能比 DNN 更適合小型表格資料 |
| [HW2](ml2021/hw02-phoneme-classification/README.md) | 39-class phoneme classification | Private accuracy `0.73610` | Activation、regularization、normalization 與 scheduler 的控制實驗 |
| [HW3](ml2021/hw03-food11-classification/README.md) | Food-11 image classification | Private accuracy `76.748%` | Augmentation、CNN、pseudo-label 與消融比較 |
| [HW4](ml2021/hw04-speaker-classification/README.md) | 600-class speaker classification | Private accuracy `0.91000` | Self-Attention、sequence length 與 Attention Pooling |
| [HW5](ml2021/hw05-machine-translation/README.md) | English-to-Traditional-Chinese translation | Local validation BLEU `23.59` | GRU／LSTM／Transformer、SentencePiece、BLEU 與 GPU smoke test |

五個作業皆保留任務、資料流程、模型、Loss、Optimization、實驗結果、失敗原因、限制與重現方式。資料集、模型權重、checkpoint 與 submission 不放入 GitHub。

### 2. LeetCode C++：26 題

| 題型 | 題數 | 題號 |
|---|---:|---|
| String／Two Pointers | 5 | 28、125、392、80、167 |
| Hash Table | 9 | 383、205、290、242、1、202、219、49、128 |
| Linked List | 6 | 138、141、21、2、92、19 |
| Binary Tree／Expression Parsing | 6 | 104、100、224、101、112、105 |
| **合計** | **26** | 詳見 [LeetCode 學習紀錄](leetcode-cpp/README.md) |

主要收穫是從題目資料關係辨認合適方法，例如雙指標、hash mapping、dummy node、Floyd cycle detection、tree recursion 與 traversal reconstruction，而不是只背單一模板。

### 3. C++ 助教準備

- 完成 CH1～CH4 期中前教學攻略。
- 整理 C++ 基本結構、輸入輸出、型別、運算式、條件判斷與 `switch`。
- 建立 Input → Process → Output 的解題教學路線。
- 整理整數除法、`=`／`==`、區間判斷、`if` 後分號與 `switch break` 等常見錯誤。
- 建立一般值、邊界值與錯誤輸入的測試觀念。

本期完成範圍停在 CH1～CH4；CH5 之後不列為暑期已完成成果。

### 4. YOLO 與專題課教材

- 在 Windows CPU-only 環境完成 pretrained YOLO 圖片批次推論。
- 六張測試圖片共偵測 8 隻鳥，保存 confidence、bounding box 與計數流程。
- 完成 Raspberry Pi × OpenCV × YOLO 四小時教材包。
- 教材包含課前驗收、教師講稿、學生手冊、投影片內容、starter、教師完成版、故障排除與離線備案。
- 清楚區分 classification、detection、segmentation，以及 confidence、IoU、NMS 的角色。
- 暑期後段實際完成 Raspberry Pi 專題課授課，將教材由準備成果轉為真實教學成果。
- 建立公開的 [raspberry-pi-yolo-course](https://github.com/warlock-hhh/raspberry-pi-yolo-course) 教材 repository，讓學生直接以 `git clone` 取得一致版本，減少現場傳檔、路徑與版本混亂。
- 公開教材包含 `check_environment.py`、`first_detection.py`、`object_counter.py`、`requirements.txt` 與 10 張課堂練習圖片。
- 學生流程固定為 clone → 建立 `.venv` → 安裝 requirements → 環境檢查 → 第一次偵測 → 物件統計 → 調整 confidence threshold。

現有紀錄能確認教材與授課均完成，但沒有保存每台 Raspberry Pi 的完整驗收輸出、推論時間與記憶體量測，因此不宣稱所有設備具有一致效能。

## 五、未完成項目與限制

### Raspberry Pi 授課紀錄

Raspberry Pi 專題課已實際完成授課，但 repository 沒有完整保存以下工程紀錄：

- 每台 Raspberry Pi 的系統、Python 與套件版本
- `check_environment.py`、`first_detection.py`、`object_counter.py` 的逐台驗收輸出
- 每台設備的推論時間、記憶體與 CPU 使用量
- 課堂中遇到的設備差異、故障數量與處理結果

因此結案判定為「授課完成」，而不是「具備完整硬體 benchmark 與逐台驗收報告」。

### LeetCode 工程紀錄

題目已在網站完成並留下方法摘要，但沒有為全部 26 題保存：

- 本機可編譯 C++ 原始碼
- 系統化邊界測試
- 錯題原因
- 隔天不看答案重寫結果

### 評估限制

- HW5 的 JudgeBoi 已失效，`23.59` 是固定本機 validation split 的 BLEU，不是官方 hidden-test 分數。
- YOLO 使用 pretrained inference，沒有重新訓練，也不能由六張圖片推論整體準確率。
- Kaggle 分數與 validation 結果只能代表各作業資料與當時實驗設定。

## 六、最重要的技術收穫

1. **先建立 baseline，再增加複雜度。** HW1 的 Ridge 優於 DNN，證明模型選擇應配合資料型態。
2. **一次改一個主要變因。** HW2～HW4 的實驗表讓 activation、regularization、架構與資料長度的效果可以被解釋。
3. **資料流程和評估方式與模型同樣重要。** Split、normalization、pseudo-label threshold、BLEU tokenization 都會改變結果可信度。
4. **Smoke test 是長時間訓練前的必要關卡。** HW5 在一個真實 batch 上發現異常 loss，避免錯誤設定持續數小時。
5. **授課完成與工程驗收紀錄是兩件事。** 本次已完成 YOLO 專題課實際授課，但日後應同步保存逐台環境、輸出與效能紀錄。
6. **公開作品要能重現，也要能安全發布。** 每個公開 repo 都排除資料與權重，並使用 Secret Scanning、Push Protection 與提交前掃描。

## 七、作品集索引

- [四期學習週報](weekly-reports/)
- [Machine Learning Portfolio](ml2021/README.md)
- [LeetCode C++](leetcode-cpp/README.md)
- [C++ Teaching Preparation](cpp-ta/README.md)
- [YOLO Project Course](project-course-day3-yolo/README.md)
- [Raspberry Pi YOLO 學生教材包](https://github.com/warlock-hhh/raspberry-pi-yolo-course)
- [學習 Roadmap](docs/roadmap.md)

獨立 ML repositories：

- [warlock-hhh/ml2021-hw2](https://github.com/warlock-hhh/ml2021-hw2)
- [warlock-hhh/ml2021-hw3](https://github.com/warlock-hhh/ml2021-hw3)
- [warlock-hhh/ml2021-hw4](https://github.com/warlock-hhh/ml2021-hw4)
- [warlock-hhh/ml2021-hw5](https://github.com/warlock-hhh/ml2021-hw5)
- [warlock-hhh/raspberry-pi-yolo-course](https://github.com/warlock-hhh/raspberry-pi-yolo-course)

## 八、結案與後續移交

本計畫於 2026/09/07 當週結束；本 repository 自 2026/09/20 補登結案文件後，作為已結案的暑期學習作品集，不再把新的研究室工作、正式課程內容或其他長期計畫混入本專案。

若後續重新進行 Raspberry Pi 實機部署、擴充程式設計助教內容或發展新的 ML 題目，應在對應的新 repository 建立新的目標、環境、驗收條件與進度紀錄。本結案報告保留當時真正完成的成果與限制，作為履歷、面試與未來複習的依據。
