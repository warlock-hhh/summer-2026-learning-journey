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

## 2026/07/20–2026/08/02：Hash Table

| 題號 | 題目 | Hash Table 用途 | 時間 | 空間 |
|---:|---|---|---:|---:|
| 383 | Ransom Note | 統計可用字元次數 | O(n+m) | O(1) |
| 205 | Isomorphic Strings | 建立雙向字元映射 | O(n) | O(1) |
| 290 | Word Pattern | 建立 pattern 與 word 的雙射 | O(n) | O(n) |
| 242 | Valid Anagram | 比較字元頻率 | O(n+m) | O(1) |
| 1 | Two Sum | 儲存數值與索引，查找 complement | O(n) | O(n) |
| 202 | Happy Number | 記錄已出現的平方和，偵測循環 | O(log n) 每輪 | O(log n) |
| 219 | Contains Duplicate II | 保存數值最後出現位置 | O(n) | O(n) |
| 49 | Group Anagrams | 以排序字串或頻率向量作為 key | O(nk log k) | O(nk) |
| 128 | Longest Consecutive Sequence | 用 set 判斷序列起點並向後延伸 | O(n) | O(n) |

### 複習重點

- Hash table 的核心是把「重複搜尋」改成平均 O(1) 查找。
- 題目 205、290 要維持雙射，只檢查單方向映射會漏掉多對一錯誤。
- 題目 1 應先查 complement 再插入目前元素，避免同一索引被使用兩次。
- 題目 219 儲存最後索引即可，不需要保存每個元素的所有位置。
- 題目 128 只從 `x - 1` 不存在的數字開始走，才能維持整體 O(n)。
- 題目 202 除了 `unordered_set`，也可以用 Floyd cycle detection 將額外空間降為 O(1)。

## 2026/08/03–2026/08/16：Linked List

| 題號 | 題目 | 核心方法 | 時間 | 額外空間 |
|---:|---|---|---:|---:|
| 138 | Copy List with Random Pointer | Hash map 建立舊節點到新節點的映射 | O(n) | O(n) |
| 141 | Linked List Cycle | Floyd 快慢指標偵測環 | O(n) | O(1) |
| 21 | Merge Two Sorted Lists | Dummy node 串接較小節點 | O(n+m) | O(1) |
| 2 | Add Two Numbers | 同步走訪兩串列並維護 carry | O(max(n,m)) | O(1)* |
| 92 | Reverse Linked List II | Dummy node + 區間原地反轉 | O(n) | O(1) |
| 19 | Remove Nth Node From End of List | Dummy node + 固定距離雙指標 | O(n) | O(1) |

`*` 不計輸出串列所需空間。

### 複習重點

- Dummy node 能統一處理刪除或修改 head 的情況，減少邊界分支。
- 快慢指標不只用於找環；維持固定距離也能一次走訪找到倒數第 n 個節點。
- 題目 92 的關鍵是保存反轉區段前一節點、區段第一節點與區段後一節點，最後正確接回。
- 題目 2 必須處理兩串列長度不同以及最後仍有 carry 的情況。
- 題目 138 除了 hash map，亦可用節點交錯法把額外空間降至 O(1)，但指標操作較複雜。
- Linked List 題目應先畫出指標變化，確認每次修改 `next` 前已保存後續節點。

## 2026/08/17–2026/08/30：Binary Tree and Expression Parsing

| 題號 | 題目 | 核心方法 | 時間 | 額外空間 |
|---:|---|---|---:|---:|
| 104 | Maximum Depth of Binary Tree | DFS 遞迴回傳左右子樹最大深度 | O(n) | O(h) |
| 100 | Same Tree | 同步遞迴比較結構與節點值 | O(n) | O(h) |
| 224 | Basic Calculator | Stack／遞迴解析括號、正負號與數字 | O(n) | O(n) |
| 101 | Symmetric Tree | 鏡像遞迴比較外側與內側子樹 | O(n) | O(h) |
| 112 | Path Sum | DFS 累減剩餘目標，到 leaf 判斷 | O(n) | O(h) |
| 105 | Construct Binary Tree from Preorder and Inorder Traversal | Preorder 決定 root，inorder 切分子樹 | O(n) | O(n) |

### 複習重點

- Tree recursion 先定義函式代表什麼，再處理空節點與 leaf 的 base case。
- 題目 100 比較「相同位置」；題目 101 比較「鏡像位置」，遞迴配對方向不同。
- 題目 112 必須在 leaf 才能確認 root-to-leaf path sum，不能在中途提早成功。
- 題目 105 應用 hash map 保存 inorder 索引，避免每層線性搜尋造成 O(n²)。
- 題目 224 不屬於樹題；核心是 expression state、括號與 sign 的保存與還原。

其中 `h` 為樹高；平衡樹約 O(log n)，最差退化樹為 O(n)。
