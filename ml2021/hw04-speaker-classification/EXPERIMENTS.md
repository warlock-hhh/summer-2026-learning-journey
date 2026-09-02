# HW4 Self-Attention 實驗計畫

## 實驗原則

每次只改一個主要變因，其他設定、資料切分與 seed 保持相同。先用 validation accuracy 選模型，不要每一組都提交 Kaggle。

每次執行會在 `experiments/<實驗名稱>/` 產生：

- `experiment_config.json`：本次模型與訓練設定
- `training_history.csv`：每個 epoch 的 loss、accuracy、learning rate
- `loss_curve.png`、`accuracy_curve.png`
- `model.ckpt`：validation accuracy 最佳的模型
- `output.csv`：訓練結束後，由最佳模型自動產生的 Kaggle 提交檔

若某次只想訓練、不產生提交檔，可在命令最後加上 `--no-infer-after-train`。

## Attention Pooling 實驗

固定目前最佳的 Encoder、head、segment 與 epoch，只把 Mean Pooling 改成可學習的 Attention Pooling：

```powershell
.\.venv\Scripts\python.exe .\hw04.py train --run-dir .\experiments\L2_H4_S256_E30_ATTNPOOL --epochs 30 --num-layers 2 --nhead 4 --segment-len 256 --batch-size 16 --pooling attention --pooling-hidden 64
```

舊 checkpoint 會自動使用 `mean`；新訓練預設使用 `attention`。若要重跑 Mean Pooling 對照組，可加上 `--pooling mean`。

## 第一輪：只比較 Encoder 層數

固定 `heads=2`、`segment=128`、`batch=16`。

```powershell
.\.venv\Scripts\python.exe .\hw04.py train --run-dir .\experiments\00_baseline_l1_h2_s128 --num-layers 1 --nhead 2 --segment-len 128 --batch-size 16

.\.venv\Scripts\python.exe .\hw04.py train --run-dir .\experiments\01_layers2_h2_s128 --num-layers 2 --nhead 2 --segment-len 128 --batch-size 16

.\.venv\Scripts\python.exe .\hw04.py train --run-dir .\experiments\02_layers3_h2_s128 --num-layers 3 --nhead 2 --segment-len 128 --batch-size 16
```

比較三組 validation accuracy，選擇最佳 `num_layers`。

## 第二輪：只比較 Attention heads

將 `<最佳層數>` 換成第一輪勝出的數字。固定 `segment=128`。

```powershell
.\.venv\Scripts\python.exe .\hw04.py train --run-dir .\experiments\10_heads1 --num-layers <最佳層數> --nhead 1 --segment-len 128 --batch-size 16

.\.venv\Scripts\python.exe .\hw04.py train --run-dir .\experiments\11_heads4 --num-layers <最佳層數> --nhead 4 --segment-len 128 --batch-size 16
```

`heads=2` 已由第一輪提供，不需要重跑。`d_model=80` 時，每個 head 的維度分別為 80、40、20。

## 第三輪：只比較語音片段長度

固定前兩輪選出的最佳層數與 heads。

```powershell
.\.venv\Scripts\python.exe .\hw04.py train --run-dir .\experiments\20_segment192 --num-layers <最佳層數> --nhead <最佳heads> --segment-len 192 --batch-size 12

.\.venv\Scripts\python.exe .\hw04.py train --run-dir .\experiments\21_segment256 --num-layers <最佳層數> --nhead <最佳heads> --segment-len 256 --batch-size 8
```

`segment=128` 已由前兩輪提供。片段變長時降低 batch size，避免 RTX 3050 4 GB 顯存不足。

## 產生指定實驗的 Kaggle CSV

```powershell
.\.venv\Scripts\python.exe .\hw04.py infer --run-dir .\experiments\<實驗名稱>
```

程式會從 checkpoint 自動讀取 `num_layers`、`nhead` 等架構，不必再次手動輸入。

## 判讀標準

優先比較：

1. 最佳 validation accuracy 是否提高。
2. Validation loss 是否穩定下降。
3. Train 與 validation accuracy 差距是否擴大（過擬合）。
4. 訓練時間及顯存是否仍可接受。

最後只將最有希望的 1～2 組提交 Kaggle，避免浪費每日提交次數。
