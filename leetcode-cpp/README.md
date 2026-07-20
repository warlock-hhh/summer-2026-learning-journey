# LeetCode C++

## 2026/07/06–2026/07/19

| 題號 | 題目 | Pattern | 關鍵判斷 |
|---:|---|---|---|
| 28 | Find the Index of the First Occurrence in a String | String matching | 固定起點後逐字比較 needle |
| 125 | Valid Palindrome | Two pointers | 忽略非英數字元並比較左右字元 |
| 392 | Is Subsequence | Two pointers | 只在字元匹配時推進 subsequence 指標 |
| 80 | Remove Duplicates from Sorted Array II | Fast/slow pointers | 目前元素和輸出位置前兩格比較 |
| 167 | Two Sum II | Left/right pointers | 和太小移左指標，和太大移右指標 |

## 複習重點

- 雙指標成立的前提通常是有序、從兩端收斂、依序匹配或原地修改。
- 題目 80 的慢指標代表「目前合法輸出區間的下一個位置」。
- 題目 167 必須利用輸入已排序，才能根據總和大小安全排除一側。
- 題目 28 的暴力解為 O(nm)，後續可用 KMP 學習如何避免重複比對。

後續每題應補上可編譯解答、邊界測試、時間／空間複雜度與隔天重寫結果。
