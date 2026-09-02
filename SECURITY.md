# Security and credential policy

本 repository 不保存任何真實 API Key、Token、密碼、私鑰或雲端 credentials。

## 開發規則

1. 憑證只放在作業系統環境變數、GitHub Actions Secrets 或未追蹤的 `.env`。
2. 可提交的 `.env.example` 只能包含欄位名稱與假值，例如 `OPENAI_API_KEY=demo`。
3. 前端公開變數（例如 `VITE_*`、`NEXT_PUBLIC_*`）不得放秘密。
4. 新增外部服務時，Key 僅授予需要的 API、資源與期限；能唯讀就不給寫入權限。
5. 提交前執行 staged scan，公開前再執行 full-history scan。

## 啟用提交前檢查

在 repository 根目錄執行一次：

```powershell
git config core.hooksPath .githooks
```

之後每次 `git commit` 都會執行：

```powershell
.\scripts\scan-secrets.ps1 -StagedOnly
```

公開或發佈前另執行：

```powershell
.\scripts\scan-secrets.ps1 -History
```

## Key 管理與事件處理

- 每個專案、環境與服務使用不同 Key，設定最低權限、使用額度與到期日。
- 有使用付費 API 時，每月至少檢查一次 usage、帳單、異常 IP／地區與失敗請求。
- 長期 Key 依服務風險定期輪替；人員、設備或專案結束時立即撤銷。
- Key 若出現在 commit、log、截圖或公開訊息中，視為已洩漏：先撤銷舊 Key、檢查使用紀錄與帳單，再建立新 Key。只刪除檔案或 commit 文字並不足夠。
- 發現問題時不要在公開 Issue 貼出憑證內容，應直接撤銷並透過 GitHub 帳號提供的私人聯絡方式回報。

目前此學習專案沒有使用需要 API Key 的外部服務；這份規範是預防未來新增整合時誤提交憑證。
