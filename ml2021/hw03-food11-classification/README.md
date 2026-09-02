# HW3：Food-11 Image Classification

從基礎 CNN 逐步加入資料增強、regularization、GAP、Residual、CutMix、ensemble 與 pseudo-label，最後以五層 CNN 搭配 dynamic self-labeling，達到 validation `76.36%`、Kaggle Public／Private `77.897%`／`76.748%`。

## 我學到的事

- 小型影像資料的瓶頸常在泛化能力，augmentation 必須符合影像語意。
- pseudo-label 要等 teacher 足夠可靠後再啟動，並用 confidence threshold 控制雜訊。
- Ensemble、TTA、CutMix 並非必然提升；需要固定 validation protocol 做消融比較。

## 重現

1. 依 [ML2021 HW03](https://github.com/ga642381/ML2021-Spring/tree/main/HW03) 規範取得受限制的 Food-11 作業資料，解壓為 `food-11/`。
2. 安裝環境後先跑學習版或 baseline，再跑最終版：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python hw03_learning.py --show-shapes
.\.venv\Scripts\python hw03_baseline.py --epochs 1
.\.venv\Scripts\python hw03_reference.py --epochs 500
```

程式支援的完整參數請用 `--help` 查看。資料、checkpoint 與 prediction CSV 已排除。完整消融分析見 [實驗技術解析](HW3_實驗技術完整解析.md)。

## 來源

題目來自李宏毅老師 ML2021 HW03；公開內容保留自行整理的可執行版本、實驗流程與結果，不重新散布課程限定資料。
