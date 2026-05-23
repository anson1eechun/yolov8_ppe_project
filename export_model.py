"""
導出模型為不同格式（ONNX、TensorRT 等）
"""

from ultralytics import YOLO
import os

MODEL_PATH = "models/best.pt"


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：找不到模型檔案 {MODEL_PATH}")
        return

    print(f"載入模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("\n可用的導出格式：")
    print("1. ONNX")
    print("2. TensorRT")
    print("3. CoreML")
    print("4. TensorFlow SavedModel")
    print("5. TensorFlow Lite")
    print("6. OpenVINO")
    print("7. 全部格式")

    choice = input("\n請選擇要導出的格式 (1-7，直接按 Enter 預設為 ONNX): ").strip()

    formats = {
        "1": "onnx",
        "2": "engine",
        "3": "coreml",
        "4": "saved_model",
        "5": "tflite",
        "6": "openvino",
        "7": "all",
    }

    export_format = formats.get(choice, "onnx")
    formats_to_export = (
        ["onnx", "engine", "coreml", "saved_model", "tflite", "openvino"]
        if export_format == "all"
        else [export_format]
    )

    print("\n開始導出模型...")
    for fmt in formats_to_export:
        try:
            print(f"\n導出為 {fmt.upper()} 格式...")
            model.export(format=fmt, imgsz=640)
            print(f"✓ {fmt.upper()} 格式導出成功")
        except Exception as e:
            print(f"✗ {fmt.upper()} 格式導出失敗: {e}")

    print("\n導出完成！")


if __name__ == "__main__":
    main()
