# HW1：COVID-19 Cases Prediction

以表格特徵預測隔日陽性案例數，並用相同 validation split 比較 mean baseline、Ridge、Random Forest 與 DNN。最終 Kaggle Public／Private RMSE 為 `0.89396`／`0.92892`。

## 我學到的事

- 小型 tabular data 不保證 DNN 優於線性模型；先建立簡單 baseline 才知道複雜模型是否真的有價值。
- StandardScaler 必須只在 training split 上 `fit`，validation/test 只能 `transform`，避免資料洩漏。
- RMSE 對大誤差較敏感；模型選擇需固定 validation split 才能公平比較。

## 重現

1. 從 [ML2021 HW01](https://github.com/ga642381/ML2021-Spring/tree/main/HW01) 依課程說明取得 `covid.train.csv`、`covid.test.csv` 與 `sampleSubmission.csv`，放在本目錄。
2. 建立環境並執行：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python ridge_baseline.py
```

輸出 `submission.csv`，此檔已由 `.gitignore` 排除。完整比較與失敗嘗試見 [研究日誌](HW1_研究日誌.md)。

## 來源

題目與資料來自李宏毅老師 ML2021 Spring HW01；`ridge_baseline.py` 是本次用來比較傳統模型的實驗程式。
