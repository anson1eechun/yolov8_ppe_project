"""
測試訓練好的模型
對測試集進行評估並顯示結果
"""

from ultralytics import YOLO
import os

MODEL_PATH = "models/best.pt"
DATA_YAML = "PPE detection.v1i.yolov8/data.yaml"
CLASS_NAMES = [
    "boots", "gloves", "goggles", "helmet", "no-boots",
    "no-gloves", "no-goggles", "no-helmet", "no-vest", "vest",
]


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：找不到模型檔案 {MODEL_PATH}")
        return

    print(f"載入模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("\n正在評估測試集...")
    metrics = model.val(data=DATA_YAML, split="test")

    print("\n=== 測試集評估結果 ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"精確度 (Precision): {metrics.box.mp:.4f}")
    print(f"召回率 (Recall): {metrics.box.mr:.4f}")

    print("\n=== 各類別詳細結果 ===")
    if hasattr(metrics, "box") and hasattr(metrics.box, "maps"):
        for name, map50 in zip(CLASS_NAMES, metrics.box.maps):
            print(f"{name:15s} mAP50: {map50:.4f}")


if __name__ == "__main__":
    main()
