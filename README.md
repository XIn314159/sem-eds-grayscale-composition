# sem-eds-grayscale-composition

Python tool for converting SEM grayscale images into EDS-calibrated composition maps, with LOOCV error estimation.

以 EDS 點分析結果校正 SEM 影像的灰階值，將整張背向散射電子（BSE）影像轉換為連續的成分分布圖，並用留一交叉驗證（LOOCV）與逐像素預測區間評估結果的可靠程度。

---

## 這個工具做什麼

SEM 的 BSE（背向散射電子）影像，灰階反映的是局部平均原子序（Z-contrast），但 EDS 只能逐點量測成分。要看整張影像的成分分布，通常只能靠人眼比對灰階深淺，而人眼可分辨的灰階層級有限，逐點比對也相當耗時。

這個工具做的事情是：

1. 以影像上已標記 EDS 量測位置的黃色十字為校正點，建立「灰階值 ↔ EDS 成分比例」的校正關係。
2. 用這條關係把整張影像的灰階換算成連續的成分分布圖。
3. 同時回報這個換算「可信到什麼程度」：LOOCV 誤差、逐像素預測區間，以及超出校正範圍的外插區域。

除成分估計外，程式也會計算各灰階類別的面積占比（Part A），可用來量化影像中不同灰階區域所占的比例，即使沒有 EDS 數據也能單獨執行。程式內建以某兩種元素的比例（如 Ti/(Ti+Fe)）作為校正目標，只要 EDS 表格提供對應欄位，即可套用在不同影像上。

## 方法流程

```
SEM 影像（含黃色十字＋綠色點號標籤）        EDS 匯出 .xlsx
        │                                        │
        ▼                                        ▼
 ① 偵測黃色十字位置                       ④ 讀取每點對應元素含量，
   （min(R,G) − B 的相對黃度，                計算元素比例，
    對 JPEG 壓縮造成的褪色具穩健性）           重複量測的點取平均
        │                                        │
        ▼                                        │
 ② OCR 讀取十字左上方的綠色點號標籤               │
   （失敗時可用 MANUAL_POINT_OVERRIDE 手動指定）  │
        │                                        │
        ▼                                        │
 ③ 取十字周圍小區域的灰階中位數                    │
   （排除十字本身與標籤文字等有色像素）             │
        │                                        │
        └──────────────┬─────────────────────────┘
                       ▼
        ⑤ 建立分析遮罩：排除黑色背景、儀器資訊列、比例尺、標記像素
           · 背景 = 灰階低於直方圖第一個谷值，且「與影像邊界相連」
             （影像內部被包圍的孔隙／裂隙不會被誤判為背景）
           · 資訊列／比例尺 = OCR 偵測文字＋亮線偵測，
             失敗時改為手動指定底部邊界
                       │
                       ▼
        ⑥ 線性校正：元素比例 = a × gray + b
           回報 R²、p 值、樣本內 RMSE、LOOCV RMSE
                       │
                       ▼
        ⑦ 輸出面積占比統計、成分分布圖、校正曲線圖、
           逐像素 95% 預測區間圖、外插區域標示
```

| 指標 | 意義 | 何時參考 |
|---|---|---|
| 樣本內 RMSE | 擬合線對「用來擬合的點」本身的誤差 | 樣本數少時容易偏樂觀，不宜單獨引用 |
| **LOOCV RMSE** | 每次拿掉一個校正點、用其餘點重新擬合並預測被拿掉的點，對所有點取 RMSE | **建議引用的預測誤差**，反映對未參與擬合資料的表現 |
| 95% 預測區間半寬 | 每個像素估計值的不確定範圍，由殘差散布，以及該灰階距校正平均灰階的遠近決定 | 判斷兩個區域的差異是否超過不確定度 |

若 LOOCV RMSE 遠大於樣本內 RMSE（程式以 2 倍為警示門檻），代表擬合線高度依賴特定幾個點，有過度擬合風險，程式會自動印出警告；若 p > 0.05，也會提醒灰階與成分的關係在統計上不顯著，不宜把結果當成已驗證的成分量測。

### 範例影像的驗證結果

| 項目 | 數值 |
|---|---|
| 校正點數 n | 〔執行後由 `analysis_metadata.json` 的 `calibration.n_points` 填入〕 |
| 校正範圍（元素比例） | 〔`calibration.ti_ratio_range`〕 |
| R² / p 值 | 〔`calibration.r2` / `calibration.p_value`〕 |
| 樣本內 RMSE | 〔`calibration.rmse_in_sample` × 100，個百分點〕 |
| **LOOCV RMSE** | 〔`calibration.loocv_rmse` × 100，個百分點〕 |
| 預測區間半寬中位數 | ± 〔`calibration.median_pi_halfwidth_pct`〕 個百分點 |
| 外插像素比例 | 〔`calibration.extrapolated_percent`〕 % |

*這些數值會在執行後自動寫入 `analysis_metadata.json`，把對應的欄位填進上表即可，不需要另外計算。*

### 限制

這個工具給出的是「以 EDS 校正的灰階估計」，不是成分的直接量測，使用時請注意：

1. **校正點很少。** 一張影像通常只有數個校正點，只涵蓋有限的成分與灰階範圍。程式在少於 5 個校正點時會明確警告，這只是流程示範，而非可辯護的校正。
2. **校正只適用於單張影像。** 灰階取決於該次 SEM 的亮度、對比度與工作條件，不同影像不能共用同一條校正線。
3. **假設灰階只反映成分。** 未拋光的樣品會有形貌造成的對比（邊緣、傾斜、陰影），這些會混入同一個灰階訊號而無法分離。
4. **外插區域不可信。** 灰階超出校正範圍的像素，校正對它們沒有任何約束，圖上以灰色標示而不給顏色。
5. **有效樣本數是校正點數，不是像素數。** 統計檢定的自由度來自校正點，不會因為影像有數百萬個像素而增加，也不會因為成分圖看起來平滑就更可信。
6. **空間解析度不匹配。** EDS 的作用體積（約 1 µm 或更大）遠大於單一像素，點分析得到的是一個範圍的平均組成。
7. **線性關係是簡化假設。** 只有在校正範圍內近似成立，才適合這樣換算。

建議在報告中將結果表述為「EDS-calibrated grayscale estimate」，並同時附上 LOOCV RMSE，而不是單獨引用 R²。

## 如何執行

### 環境需求

- Python 3.9 以上
- 套件：見 `requirements.txt`（`opencv-python`、`numpy`、`pandas`、`matplotlib`、`scipy`、`Pillow`、`openpyxl`；`pytesseract` 為選用）
- **Tesseract OCR**（選用）：用於自動讀取十字旁的點號、偵測儀器資訊列與比例尺。未安裝或安裝失敗時，程式會自動退回手動模式：資訊列由使用者輸入要排除的像素高度，點號需在腳本開頭的 `MANUAL_POINT_OVERRIDE` 中手動指定。
  - Windows 安裝後，程式會依序在系統常見安裝路徑與 PATH 中尋找。
  - 需要有 `tessdata/eng.traineddata` 語言資料，光有執行檔會報錯。

```bash
pip install -r requirements.txt
```

### 輸入檔案

| 檔案 | 要求 |
|---|---|
| SEM 影像（png／jpg） | 以黃色「＋」標記 EDS 量測位置，每個十字左上方有綠色的點號標籤；下方通常有儀器資訊列或比例尺 |
| EDS 匯出表（.xlsx） | 需包含 `Spectrum` 及對應的元素欄位；`Spectrum` 欄以 `<編號>.spx` 結尾；同一點有多次量測時自動取平均 |

EDS 表可省略（直接按 Enter 略過），此時只會執行 Part A 的面積占比分析，不做成分校正。

### 執行

```bash
python src/sem_analysis.py
```

程式會依序詢問：

1. 影像路徑（可直接把檔案拖進終端機視窗）
2. EDS 表路徑（可留空跳過）
3. 輸出資料夾名稱（預設沿用影像檔名）
4. 灰階類別數（預設等於偵測到的十字數）
5. 背景灰階門檻（預設為自動偵測的直方圖谷值，可覆寫）
6. 若 OCR 沒有把資訊列或比例尺排除乾淨，可再補一段底部要排除的像素高度

### 輸出檔案

結果存放在腳本旁的 `SEMANALYSIS result/<輸出資料夾名稱>/`：

| 檔案 | 內容 |
|---|---|
| `detected_crosses.png` | 十字偵測結果、灰階值與資訊列遮罩，用來人工檢查偵測是否正確 |
| `analysis_mask.png` | 實際納入計算的像素（白色＝前景，其餘為背景／資訊列／標記） |
| `histogram.png`、`classified_image.png` | 灰階直方圖與各類別的面積占比、分類地圖 |
| `results.csv` | 各灰階類別的像素數、面積占比與對應 EDS 成分（Part A） |
| `calibration_points.csv`、`calibration_fit.png` | 校正點資料與校正曲線（Part B） |
| `composition_map.png` | 校正後的元素比例分布圖（外插區域以灰色標示） |
| `uncertainty_map.png` | 逐像素 95% 預測區間半寬 |
| `analysis_metadata.json` | 所有參數、統計結果與校正細節，方便重現與引用 |
| `detected_crosses_debug.png` | 只有點號比對失敗時才會產生，顯示各十字的放大圖與 OCR 讀值 |

### 常見問題

- **點號讀錯或重複：** 程式偵測到同一個點號對應多個十字時會停止校正。打開 `detected_crosses_debug.png`，依圖上標示的座標 `key=(x, y)` 填入腳本開頭的 `MANUAL_POINT_OVERRIDE`，例如 `{(631, 283): 1, ...}`。
- **背景判斷不對：** 確認 `detected_crosses.png` 上青色的 `BK` 標記落在黑色背景上；否則在提示時手動輸入背景門檻。
- **`eng.traineddata` 找不到：** 把語言資料放在名為 `tessdata` 的資料夾中，位於 Tesseract 執行檔旁，或設定 `TESSDATA_PREFIX` 環境變數。

## 專案結構

```
sem-eds-grayscale-composition/
├── README.md
├── requirements.txt
├── LICENSE
├── src/
│   └── sem_analysis.py
└── results/
    ├── composition_map.png
    └── calibration_fit.png
```

## License

MIT License。
