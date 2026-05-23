"""
使用訓練好的模型進行預測
支援單張圖片、圖片資料夾、MP4 影片檔案
"""

from ultralytics import YOLO
import os
import sys

MODEL_PATH = "models/best.pt"


def is_video_file(filepath):
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]
    return any(filepath.lower().endswith(ext) for ext in video_extensions)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：找不到模型檔案 {MODEL_PATH}")
        print("請先執行 train.py 訓練模型，或確認 models/best.pt 存在")
        return

    print(f"載入模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        print("請提供輸入檔案路徑")
        print("\n用法:")
        print("  圖片: py predict.py <圖片路徑>")
        print("  影片: py predict.py <影片路徑.mp4>")
        print("  資料夾: py predict.py <資料夾路徑>")
        return

    if not os.path.exists(source):
        print(f"錯誤：找不到檔案或資料夾 {source}")
        return

    is_video = is_video_file(source) if os.path.isfile(source) else False

    if is_video:
        print(f"\n偵測到影片檔案: {source}")
        print("開始處理影片...")
    else:
        print(f"\n正在對 {source} 進行預測...")

    results = model.predict(
        source=source,
        save=True,
        conf=0.25,
        save_txt=True,
        save_conf=True,
        show=False,
        line_width=2,
    )

    if is_video:
        print("\n✓ 影片處理完成！")
        print("結果影片儲存在: runs/detect/predict/")
    else:
        print("\n✓ 預測完成！")
        print("結果儲存在: runs/detect/predict/")
        print(f"共處理 {len(results)} 個檔案")


if __name__ == "__main__":
    main()
