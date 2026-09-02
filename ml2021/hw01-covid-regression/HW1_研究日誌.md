# ML2021Spring HW1 研究日誌

## 作業目標

本次作業是 Kaggle competition `ML2021Spring-hw1`，題目為 COVID-19 Cases Prediction。任務屬於 regression，目標是根據訓練資料中的州別、問卷特徵與過去確診相關欄位，預測測試資料中的 `tested_positive`。

Kaggle 評分方式為 RMSE，分數越低代表預測越接近隱藏答案。提交檔案不是程式碼，而是 CSV 預測結果，格式如下：

```csv
id,tested_positive
0,預測值
1,預測值
...
```

## 資料理解

本次資料包含：

- `covid.train.csv`：2700 筆資料，含 target，共 95 欄
- `covid.test.csv`：893 筆資料，不含 target，共 94 欄
- `sampleSubmission.csv`：Kaggle 要求的提交格式，共 893 筆

訓練資料比測試資料多出的欄位是：

```text
tested_positive.2
```

因此在本地訓練時將其視為 target：

```python
target = "tested_positive.2"
```

而 Kaggle submission 的預測欄位名稱則是：

```text
tested_positive
```

## Baseline 建立

一開始先建立簡單的 sklearn baseline，目的是確認整個 ML 流程可以跑通：

```text
讀資料 -> 拆 X/y -> train/validation split -> 訓練模型 -> 算 RMSE -> 產生 submission
```

使用模型：

- Ridge Regression
- RandomForestRegressor

資料切分方式：

```python
train_test_split(test_size=0.2, random_state=42)
```

其中：

- `X_train, y_train` 用於訓練
- `X_valid, y_valid` 用於本地驗證
- `X_test` 用於產生 Kaggle submission

## 實驗紀錄

| 版本 | 方法 | 主要設定 | Private RMSE | Public RMSE | 結論 |
|---|---|---|---:|---:|---|
| v0 | sklearn Ridge 全特徵 | `Ridge(alpha=10)`，使用全部特徵 | 0.92892 | 0.89396 | 目前最佳 baseline |
| v1 | PyTorch DNN 原始 sample code | 全 93 特徵，Adam | 2.53401 | 2.58992 | 泛化很差 |
| v2 | PyTorch DNN feature selection | 40 states + 2 tested_positive | 1.23853 | 1.23634 | 特徵選擇明顯改善 DNN |
| v3 | PyTorch DNN 修正 normalization | dev/test 使用 train mean/std | 1.20425 | 1.21625 | 有小幅改善 |
| v4 | DNN 調 learning rate + weight decay | `lr=0.0005`, `weight_decay=1e-5` | 1.23944 | 1.22686 | 沒改善，private 變差 |
| v5 | DNN 加深模型 | hidden layers: 64 -> 32 | 1.26009 | 1.29609 | 模型變複雜後變差 |
| v6 | DNN 改 batch size | `batch_size=64` | 1.21508 | 1.21118 | public 小幅改善，但 private 輸 v3 |
| v7 | sklearn Ridge feature selection | 40 states + 2 tested_positive | 0.95019 | 0.95100 | 對 Ridge 反而變差 |

## 重要觀察

### 1. 簡單模型不一定比較差

Ridge Regression 雖然是線性模型，但在本題表現最好。推測原因是本題資料量小，且 target 與過去的 `tested_positive` 欄位存在高度線性關係，因此線性模型反而泛化穩定。

### 2. DNN 不一定適合小型 tabular data

PyTorch DNN 雖然模型能力較強，但在本題的表格資料上容易 overfit。特別是使用全部 93 個特徵時，Kaggle 分數明顯變差，表示模型可能學到雜訊。

### 3. Feature selection 對 DNN 有幫助

助教提示的 feature selection：

```python
feats = list(range(40)) + [57, 75]
```

代表使用：

```text
40 個州別特徵 + tested_positive + tested_positive.1
```

這使 DNN 分數從約 2.53 改善到約 1.24，顯示過去確診資料對預測最關鍵。

### 4. Feature selection 對 Ridge 沒有幫助

同樣的 feature selection 套用到 Ridge 後，分數從：

```text
0.92892 / 0.89396
```

變成：

```text
0.95019 / 0.95100
```

代表 Ridge 可以有效利用更多特徵，且正則化能降低雜訊特徵的影響。

### 5. 修正 normalization 有效但有限

sample code 原本讓 train/dev/test 各自計算 mean/std，較不符合標準 ML 流程。修正為只用 train set 的 mean/std 後，DNN private score 從 1.23853 改善到 1.20425，但仍未超越 Ridge。

## 技術概念整理

### RMSE

RMSE 是 Kaggle 使用的評分指標：

```text
RMSE = sqrt(MSE)
```

它衡量預測值與真實值之間的平均誤差，越低越好。

### Validation / Dev set

`valid` 和 `dev` 是同一個概念，都是從 training data 切出來的小考資料，用於估計模型在未見資料上的表現。

### Baseline

Baseline 是第一個簡單可用的模型，用來當後續實驗比較基準。本次最重要的 baseline 是 sklearn Ridge 全特徵版本。

### Adam 與 Momentum

DNN sample code 使用 PyTorch optimizer。原始助教 code 使用 SGD + momentum，後來改用 Adam。Adam 內部已經包含類似 momentum 的一階與二階動量，因此不需要另外加入 `momentum` 參數。

## 目前最佳結果

目前最佳提交為：

```text
sklearn Ridge Regression 全特徵
Private RMSE: 0.92892
Public RMSE: 0.89396
```

這份結果雖然不是排行榜頂尖，但已經是一個穩定且合理的 baseline，也優於目前調整過的 DNN sample code。

## Meeting 可報告重點

1. 本題是 regression，提交的是 test set 的 `tested_positive` 預測值，不是提交 RMSE。
2. 先用 sklearn Ridge 建立 baseline，成功取得 private 0.92892 / public 0.89396。
3. 助教 PyTorch DNN sample code 原始版本分數很差，顯示 DNN 不一定適合小型 tabular data。
4. 依助教 TODO 做 feature selection 後，DNN 有明顯改善，但仍未超越 Ridge。
5. 修正 normalization 後 DNN 小幅改善，代表資料前處理會影響泛化。
6. 調 learning rate、weight decay、hidden layers、batch size 後，皆未超越 Ridge。
7. 結論：本題目前以 Ridge 全特徵作為最佳模型，DNN 則作為理解 PyTorch 訓練流程的練習。

## 後續方向

若要繼續提升分數，可以往以下方向嘗試：

- 使用 sklearn 的 `GradientBoostingRegressor`
- 嘗試 `ElasticNet` 或 `SVR`
- 將 Ridge 與 DNN 預測做 ensemble
- 重新設計 validation split，避免 dev set 與 Kaggle test 分布不一致
- 系統化記錄每次實驗設定與 Kaggle public/private score
