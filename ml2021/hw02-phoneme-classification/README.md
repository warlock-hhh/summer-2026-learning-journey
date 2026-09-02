# HW2：TIMIT Framewise Phoneme Classification

輸入相鄰 11 個 frames 的 MFCC 特徵（`11 × 39 = 429` 維），以 MLP 預測 39 類音素。透過 activation、dropout、BatchNorm、scheduler 與模型寬度的控制實驗，最佳 Public／Private accuracy 為 `0.73628`／`0.73610`。

## 我學到的事

- ReLU 比深層 Sigmoid 更不易出現梯度飽和。
- Dropout、weight decay 與 BatchNorm 解決的問題不同，必須逐項改動才能判斷效果。
- training loss 與 validation accuracy 要一起觀察；模型變寬雖可提升 capacity，也可能擴大過擬合。

## 重現

1. 依 [ML2021 HW02](https://github.com/ga642381/ML2021-Spring/tree/main/HW02) 的規範取得 TIMIT 作業資料。
2. 將三個 `.npy` 檔放在 `timit_11/timit_11/`。
3. 執行：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python train_phoneme_classifier.py
```

資料、權重與 submission 不納入 Git。各輪實驗與技術判讀見 [研究日誌](HW2_研究日誌.md)。

## 來源

程式由 ML2021 HW02 公開 starter notebook 修改，保留原課程標示；本目錄的參數實驗、結果與研究日誌為本次學習成果。
