#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import cv2


def main():
    # 1. モデルのロード（ここでは軽量版の yolov8n を使用）
    model = YOLO('yolo11n.pt')

    # 2. 入力画像の読み込み
    image_path = 'model/data/images/val/val_hand_0.JPG'
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    # 3. 物体検出（inference）
    #    source にファイルパスを渡すこともできます: model.predict(source=image_path, ...)
    results = model.predict(source=img, conf=0.25, save=False)

    # 4. 結果の描画と表示
    #    YOLOv8 の返り値は Result オブジェクトのリストです（通常は単一要素）
    result = results[0]

    # バウンディングボックスとクラス名を OpenCV で描画
    for box in result.boxes:
        # 箱座標は (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{model.names[cls_id]} {conf:.2f}"

        # 四角形とラベルの描画
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 5. 結果の表示
        output_path = 'result.jpg'
        cv2.imwrite(output_path, img)
        print(f"Detection result saved to {output_path}")


if __name__ == '__main__':
    main()
