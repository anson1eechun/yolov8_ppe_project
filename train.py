"""
YOLOv8n 模型訓練腳本
用於訓練 PPE (個人防護裝備) 檢測模型
"""

from ultralytics import YOLO
import os
import torch

DATA_YAML = "PPE detection.v1i.yolov8/data.yaml"
PRETRAINED = "models/yolov8n.pt"
CHECKPOINT = "runs/detect/ppe_yolov8n/weights/last.pt"


def main():
    if not os.path.exists(DATA_YAML):
        print(f"錯誤：找不到資料集配置檔案 {DATA_YAML}")
        print("請將 PPE detection.v1i.yolov8 資料夾放到專案根目錄")
        return

    if torch.cuda.is_available():
        device = 0
        batch_size = 16
        workers = 8
        print("偵測到 CUDA GPU，將使用 GPU 進行訓練")
    else:
        device = "cpu"
        batch_size = 4
        workers = 4
        print("未偵測到 CUDA GPU，將使用 CPU 進行訓練（訓練速度會較慢）")

    resume_training = os.path.exists(CHECKPOINT)

    if resume_training:
        print(f"找到檢查點，將從 {CHECKPOINT} 恢復訓練...")
        model = YOLO(CHECKPOINT)
    else:
        print("載入 YOLOv8n 預訓練模型...")
        model = YOLO(PRETRAINED)

    print("開始訓練模型...")
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=batch_size,
        name="ppe_yolov8n",
        project="runs/detect",
        device=device,
        workers=workers,
        patience=50,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        resume=resume_training,
    )

    print("訓練完成！")
    print(f"最佳模型儲存在: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
