# YOLOv8 PPE 檢測專案

純 Python 的 YOLOv8 訓練與推論專案，用於個人防護裝備（PPE）物件檢測。

## 專案結構

```
yolov8_ppe_project/
├── train.py              # 訓練模型
├── predict.py            # 圖片/影片/資料夾推論
├── predict_video.py      # 影片推論（含參數選項）
├── test_model.py         # 測試集評估
├── export_model.py       # 導出 ONNX 等格式
├── requirements.txt      # Python 依賴
├── data.yaml             # 資料集類別定義（參考用）
├── models/
│   ├── yolov8n.pt        # 預訓練權重（訓練起點）
│   └── best.pt           # 已訓練好的最佳模型
└── PPE detection.v1i.yolov8/   # 資料集（需自行複製）
    ├── data.yaml
    ├── train/
    ├── valid/
    └── test/
```

## 環境建置

```bash
py -m pip install -r requirements.txt
```

有 NVIDIA GPU 時，建議安裝 CUDA 版 PyTorch：

```bash
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 資料集

從原專案複製整個 `PPE detection.v1i.yolov8` 資料夾到本專案根目錄，或解壓 `dataset.zip`。

- **類別**：boots, gloves, goggles, helmet, no-boots, no-gloves, no-goggles, no-helmet, no-vest, vest
- **訓練集**：3,597 張
- **驗證集**：1,026 張
- **測試集**：517 張

## 使用方式

### 訓練

```bash
py train.py
```

訓練結果儲存在 `runs/detect/ppe_yolov8n/weights/`。

### 推論（展示結果）

```bash
py predict.py image.jpg
py predict.py video.mp4
py predict.py path/to/folder
```

結果影像/影片儲存在 `runs/detect/predict/`。

### 測試集評估

```bash
py test_model.py
```

### 導出模型

```bash
py export_model.py
```
