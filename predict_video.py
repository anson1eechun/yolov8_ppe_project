"""
專門用於處理影片的預測腳本
"""

from ultralytics import YOLO
import os
import sys

MODEL_PATH = "models/best.pt"


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：找不到模型檔案 {MODEL_PATH}")
        return

    print(f"載入模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    if len(sys.argv) < 2:
        print("請提供影片檔案路徑")
        print("\n用法: py predict_video.py <影片路徑.mp4> [選項]")
        print("\n選項:")
        print("  --conf <值>       信心度閾值 (0.0-1.0，預設: 0.25)")
        print("  --save-dir <路徑> 儲存目錄 (預設: runs/detect/predict)")
        return

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"錯誤：找不到影片檔案 {video_path}")
        return

    conf_threshold = 0.25
    save_dir = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--conf" and i + 1 < len(sys.argv):
            conf_threshold = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--save-dir" and i + 1 < len(sys.argv):
            save_dir = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    print(f"\n影片檔案: {video_path}")
    print(f"信心度閾值: {conf_threshold}")

    results = model.predict(
        source=video_path,
        save=True,
        conf=conf_threshold,
        save_dir=save_dir,
        line_width=2,
        show=False,
    )

    print("\n✓ 影片處理完成！")
    output_dir = save_dir or "runs/detect/predict/"
    print(f"結果影片儲存在: {output_dir}")


if __name__ == "__main__":
    main()
