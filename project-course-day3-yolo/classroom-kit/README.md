# Raspberry Pi × OpenCV × YOLO 四小時教材

正式題目：**圖片多物件偵測與分類統計系統**。

本教材以《RaspberryPi 樹莓派－Python × AI 超應用聖經》第 11 章為第一優先，沿用原章節順序與 `horse.jpg`，只更新不適合 Raspberry Pi 5／Debian 13 的舊工具。

| 課本 | 課堂新版 |
|---|---|
| virtualenvwrapper | 專案內 `.venv` |
| Thonny | VS Code |
| OpenCV 4.5 | 由目前環境安裝相容版本 |
| YOLOv3＋Darknet 237 MB | Ultralytics YOLO26n |
| 手動輸出層與 NMS | 新版 API；原理仍講解 |
| Webcam 延伸 | 移除 |

## 檔案

- `00_課前準備與驗收.md`：教師建置每台 Pi。
- `01_教師逐分鐘講稿.md`：四小時照表授課。
- `02_學生操作手冊.md`：學生照著執行。
- `03_投影片逐頁內容.md`：投影片製作依據。
- `04_故障排除與備案.md`：現場救援。
- `05_學生成果單.md`：課堂驗收。
- `project/student/`：學生分段程式。
- `project/teacher/final_project.py`：教師完成版。

`yolo-minimal` 是授課者先前的個人練習，不屬於正式教材。
