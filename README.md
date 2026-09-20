# Summer 2026 Learning Journey

淡江大學電機系碩士生的 2026 暑期學習紀錄，聚焦於：

- C++ / LeetCode
- Machine Learning
- Linux / Raspberry Pi
- YOLO Object Detection
- Programming Teaching Preparation

## Project Status

> **已結案／Archived**
>
> - 執行期間：2026/07/06–2026/09/20
> - 正式結案：2026/09/20

本計畫已完成暑期階段任務並停止持續更新。最終成果、目標達成度、未完成項目與後續移交請見 [暑期訓練計畫結案報告](FINAL_REPORT.md)。後續研究室、課程或硬體實作將建立新的專案紀錄，不回填為本次暑期成果。

## Goals

1. 完成李宏毅《機器學習 2021》40 部影片並建立每週摘要。
2. 使用 C++ 持續練習 LeetCode，記錄測試、複雜度與錯題重寫。
3. 準備程式設計（一）助教教材。
4. 準備 Raspberry Pi YOLO 專題課 Day 3 教材與備援方案。
5. 建立可重現、可說明、可除錯的工程習慣。

## Progress

| Week | ML2021 | LeetCode C++ | C++ TA | Pi / YOLO |
|---|---:|---:|---|---|
| W1–W2 | HW1、HW2 | 5 | 基礎運算與 I/O 練習 3 題 | 完成 OS 映像燒錄 |
| W3–W4 | HW3 Food-11 | 9 | 完成 CH1–CH4 期中前備課 | 本期無進度 |
| W5–W6 | HW4 Speaker Classification | 6 | 本期無新增 | YOLO 鳥類批次偵測 |
| W7–W8 | HW5 Translation | 6 | 本期無新增 | 完成 YOLO 四小時教材包 |
| 結案階段 | 本期無新增 | 本期無新增 | 本期無新增 | 完成 Raspberry Pi 專題課實際授課 |

## Final Outcomes

| Track | Final outcome | Status |
|---|---|---|
| Machine Learning | 完成 ML2021 HW1～HW5，涵蓋 Regression、Phoneme、CNN、Self-Attention、Seq2Seq／Transformer | 完成主要實作 |
| LeetCode C++ | 完成 26 題，涵蓋 Two Pointers、Hash Table、Linked List、Binary Tree 與 Expression Parsing | 完成刷題；程式碼與重寫紀錄未全部補齊 |
| C++ TA | 完成 CH1～CH4 期中前教學攻略與備課架構 | 完成本期範圍 |
| YOLO Course | 完成 Windows 圖片推論、四小時教材、公開 GitHub 教材包與實際授課 | 完成 |
| Raspberry Pi | 後段集中處理 Raspberry Pi 專題並完成授課；未完整保存每台設備的驗收與效能紀錄 | 授課完成、紀錄有限 |
| Engineering Practice | 建立可重現 README、研究日誌、Docker／requirements、GitHub Secret Scanning 與 pre-commit 憑證掃描 | 完成 |

## Repository Structure

```text
.
├─ docs/                    # Roadmap and public documentation
├─ weekly-reports/          # Weekly progress reports
├─ leetcode-cpp/            # C++ solutions and review notes
├─ ml2021/                  # Weekly ML2021 notes
├─ cpp-ta/                  # Original teaching materials written by me
└─ project-course-day3-yolo/# YOLO experiments and teaching materials
```

## Machine Learning Portfolio

ML2021 HW1～HW5 已整理為可閱讀、可重現的作品集；每個專案包含程式、實驗紀錄、結果、資料放置方式與限制說明。

- [HW1：COVID-19 Regression](ml2021/hw01-covid-regression/README.md)
- [HW2：TIMIT Phoneme Classification](ml2021/hw02-phoneme-classification/README.md)
- [HW3：Food-11 CNN and Pseudo-labeling](ml2021/hw03-food11-classification/README.md)
- [HW4：Speaker Classification with Self-Attention](ml2021/hw04-speaker-classification/README.md)
- [HW5：English-to-Traditional-Chinese Seq2Seq](ml2021/hw05-machine-translation/README.md)
- [ML 大架構複習筆記](ml2021/ML大架構複習筆記.md)

## Teaching Repository

- [Raspberry Pi YOLO Object Detection Course](https://github.com/warlock-hhh/raspberry-pi-yolo-course)：提供學生直接 `git clone` 的公開教材包，包含環境檢查、三階段程式與 10 張課堂練習圖片。

## Weekly Reports

- [Week 1–2｜2026/07/06–2026/07/19](weekly-reports/week01.md)
- [Week 3–4｜2026/07/20–2026/08/02](weekly-reports/week02.md)
- [Week 5–6｜2026/08/03–2026/08/16](weekly-reports/week03.md)
- [Week 7–8｜2026/08/17–2026/08/30](weekly-reports/week04.md)
- [Final Phase｜Raspberry Pi 專題課授課](weekly-reports/final-phase.md)
- [Final Report｜2026 Summer Closure](FINAL_REPORT.md)

## Public Repository Policy

This repository contains my original notes, experiments, teaching materials, and attributed adaptations of public course starter code. It does not publish private student information, unreleased exams, proprietary course files, passwords, API keys, model weights, or restricted datasets.
