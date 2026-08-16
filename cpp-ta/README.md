# C++ Teaching Preparation

## 第一階段練習

1. 美元轉日圓：練習 `cin`、`cout`、浮點數與匯率乘法。
2. 水力發電功率：練習常數、物理公式與 W 到 MW 的單位換算。
3. 時間與溫度公式：練習小時／分鐘轉換、浮點數除法與運算式。

## 教學時應提醒

- `minutes / 60` 若兩邊都是整數會截斷小數，應使用 `60.0`。
- 物理公式應標註每個變數和輸出單位，避免數值正確但量綱錯誤。
- 匯率與計算結果使用 `double`；時間欄位可先用 `int` 接收，再轉成浮點數。
- 測試至少包含一般值、零值與邊界值，並手算一組答案交叉驗證。

## CH1–CH4 期中前備課

已完成期中前教學攻略與教授說明版備課總覽，範圍涵蓋：

| 章節 | 教學主題 | 學生應具備的能力 |
|---|---|---|
| CH1 | 電腦、程式語言與編譯流程 | 說明 RAM／ROM、compiler、object file 與 executable |
| CH2 | C++ 基本結構、型別與 I/O | 使用 `main`、`cin`、`cout`、變數與基本資料型態 |
| CH3 | Expressions and Statements | 正確處理運算優先順序、型別轉換、整數除法與輸出格式 |
| CH4 | Selection Structures | 使用 `if`、`else if`、nested if、`switch` 與邏輯運算子 |

### 教學路線

```text
題目文字
→ 找出 Input／Process／Output
→ 決定變數型別
→ 寫公式、條件或 pseudocode
→ 實作 C++
→ 使用一般、邊界與錯誤測資驗證
```

### 期中前高頻錯誤

- `minutes / 60` 造成整數除法，應改成 `minutes / 60.0`。
- 把 `^` 誤認為次方；C++ 中應使用乘法或 `pow`。
- 混淆 assignment `=` 與 equality `==`。
- 將區間寫成 `90 <= score <= 100`，而非使用 `&&`。
- 在 `if` 後誤加分號，或在 `switch` 中漏掉 `break`。
- 未處理無效輸入、邊界值與除以零。

目前已具備從課程主線、範例題、常見錯誤、批改檢查到期中複習題型的一套完整備課架構。CH5 迴圈、CH6 函式、陣列與 OOP 不屬於本次完成範圍。

Restricted course files, exams, and student information must not be committed here.
