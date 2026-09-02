# HW5 Docker 環境

## 邊界

- Windows 保存程式、`DATA/` 與 `checkpoints/`。
- Docker image 固定 Linux、PyTorch/CUDA runtime、fairseq 與 Python 套件。
- 啟動時將本資料夾掛載為 Container 內的 `/workspace`。

## 一次性主機設定

以系統管理員 PowerShell 執行：

```powershell
wsl --install
```

重新開機後安裝 Docker Desktop，使用 WSL2 backend。Docker Desktop 的 GPU 支援只適用於 WSL2 backend。

## 建立 image

在本資料夾執行：

```powershell
docker compose build
```

## 驗證環境與 GPU

```powershell
docker compose run --rm hw5 python docker_smoke_test.py
```

成功標準：

- 顯示 RTX 3050。
- `CUDA available: True`。
- fairseq 可匯入。
- train/valid/test 行數正確。
- SentencePiece vocabulary 為 8,000。
- 完成 forward、backward、optimizer step。
- 最後顯示 `SMOKE TEST PASSED`。

## 產生 fairseq binary

Smoke test 通過後執行：

```powershell
docker compose run --rm hw5 fairseq-preprocess `
  --source-lang en `
  --target-lang zh `
  --trainpref /workspace/DATA/rawdata/ted2020/train `
  --validpref /workspace/DATA/rawdata/ted2020/valid `
  --testpref /workspace/DATA/rawdata/ted2020/test `
  --destdir /workspace/DATA/data-bin/ted2020 `
  --joined-dictionary `
  --workers 2
```

`DATA/` 被 `.dockerignore` 排除，不會被複製進 image；它透過 bind mount 在 Container 內顯示為 `/workspace/DATA`。
