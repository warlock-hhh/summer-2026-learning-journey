# HW4：Speaker Classification with Self-Attention

將 `(T, 40)` log-Mel 特徵送入 Transformer Encoder，逐步測試 encoder layers、attention heads、segment length、epochs 與 pooling。Attention Pooling 最終達到 validation `86.85%`、Kaggle Public／Private `0.91833`／`0.91000`。

## 我學到的事

- Self-Attention 能跨時間 frames 建立關係，但計算與記憶體成本約隨序列長度平方成長。
- Attention Pooling 可學習重要 frames 的權重，避免聲紋線索被 mean pooling 稀釋。
- 實驗應一次只改一項主要變因，並同時記錄 validation 與 hidden-test 表現。

## 重現

依 [ML2021 HW04](https://github.com/ga642381/ML2021-Spring/tree/main/HW04) 取得資料並解壓為 `Dataset/Dataset/`，接著：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python hw04.py check-data
.\.venv\Scripts\python hw04.py train --num-layers 2 --nhead 4 --segment-len 256 --epochs 30 --pooling attention --save-path model.ckpt
.\.venv\Scripts\python hw04.py infer --model-path model.ckpt --output-path output.csv
```

實際 CLI 選項以 `python hw04.py --help` 為準。完整架構、實驗表與限制見 [研究日誌](HW4_研究日誌.md) 與 [實驗計畫](EXPERIMENTS.md)。

## 來源

題目與資料格式來自李宏毅老師 ML2021 HW04；本程式的命令列介面、實驗配置、Attention Pooling 與研究紀錄為本次重整成果。
