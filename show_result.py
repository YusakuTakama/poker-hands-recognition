import cv2
from ultralytics import YOLO
import os
import torch  # PyTorchが暗黙的に使われるためインポートしておくと良い

# --- 設定項目 ---
# ファインチューニングした重みファイルのパス
WEIGHTS_PATH = "detection_model/checkpoints/best.pt"  # 例: "runs/train/exp/weights/best.pt"
# テストしたい画像のパス
IMAGE_PATH = "detection_model/data/images/val/val_hand_2.jpg"
# 信頼度の閾値 (これ未満の検出は無視)
CONFIDENCE_THRESHOLD = 0.25
# --- 設定項目ここまで ---


def load_yolo_model(weights_path):
    """
    指定された重みファイルからYOLOモデルをロードします。
    """
    try:
        model = YOLO(weights_path)
        print(f"モデルのロードに成功しました: {weights_path}")
        return model
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        return None


def perform_inference_and_save(model, image_path, confidence_threshold, output_image_path):
    """
    画像に対して推論を実行し、結果を指定されたパスに保存します。
    GUI表示は行いません。
    """
    if model is None:
        return

    try:
        # 画像を読み込み
        img = cv2.imread(image_path)
        if img is None:
            print(f"画像の読み込みに失敗しました: {image_path}")
            return

        # モデルで推論を実行
        results = model.predict(source=img, conf=confidence_threshold, verbose=False)

        # 推論結果を取得
        result = results[0]

        # 元の画像に検出結果を描画 (ultralyticsのplotは内部でOpenCVに依存)
        annotated_img = result.plot()  # このメソッド自体がGUI環境を期待する可能性もゼロではない

        # 結果を保存
        try:
            output_dir = os.path.dirname(output_image_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"出力ディレクトリを作成しました: {output_dir}")

            cv2.imwrite(output_image_path, annotated_img)
            print(f"検出結果を保存しました: {output_image_path}")
        except Exception as e:
            print(f"結果の保存中にエラーが発生しました: {e}")

    except Exception as e:
        # result.plot()が内部でGUI関連の初期化を試みて失敗する場合もここに到達する可能性
        print(f"推論または結果の描画/保存準備中にエラーが発生しました: {e}")
        print("もしこのエラーが 'result.plot()' 関連でGUIがないために発生している場合、")
        print("ultralyticsの描画機能がGUI非依存であるか確認するか、")
        print("手動でバウンディングボックス情報を取得してOpenCVで描画するコードに書き換える必要があるかもしれません。")


if __name__ == "__main__":
    # IMAGE_PATHを実際のテスト画像パスに設定してください。
    # 例: IMAGE_PATH = "path/to/your/actual/test_image.jpg"
    if not os.path.exists(IMAGE_PATH):
        print(f"エラー: 指定されたテスト画像パスが見つかりません: {IMAGE_PATH}")
        print("スクリプト上部の IMAGE_PATH を有効な画像ファイルパスに設定してください。")
    elif not os.path.exists(WEIGHTS_PATH):
        print(f"エラー: 指定された重みファイルパスが見つかりません: {WEIGHTS_PATH}")
    else:
        # 1. モデルのロード
        yolo_model = load_yolo_model(WEIGHTS_PATH)
        OUTPUT_IMAGE_PATH = "val_hand_2_result.jpg"
        # 画像の出力パスを設定

        # 2. 推論と結果の保存
        if yolo_model:
            perform_inference_and_save(yolo_model, IMAGE_PATH, CONFIDENCE_THRESHOLD, OUTPUT_IMAGE_PATH)
