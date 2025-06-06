from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from .hands_recognition import HandsRecognizer
from argparse import ArgumentParser
from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(description="Poker Hand Recognition")
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        dest="image",
        required=True,
        help="Path to the input image containing poker cards"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov11",
        choices=["yolov11", "yolov8"],
        help="Model type to use (yolov8 or yolov11)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 0. 設定 (必要に応じて変更してください)
    model_checkpoint_path = "classification_model/checkpoints/epoch170_step511_acc=92.31.pt"
    pretrained_model_name = "google/vit-base-patch16-224"
    num_labels = 52  # ファインチューニング時に設定したクラス数

    # 1. モデルの初期化
    # ファインチューニング時と同じ設定でモデルの骨格を読み込みます。
    model = ViTForImageClassification.from_pretrained(
        pretrained_model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True  # ファインチューニング時に設定した場合
    )

    # 2. ファインチューニング済みの重みの読み込み
    # PyTorchのチェックポイントは、モデルのstate_dict全体、または他の情報を含む辞書として保存されている場合があります。
    # 一般的には state_dict を直接ロードします。
    # もしチェックポイントが辞書形式で、その中に 'model_state_dict' や 'state_dict' のようなキーで
    # モデルの重みが保存されている場合は、適宜修正してください。
    try:
        # state_dict のみを保存した場合
        state_dict = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))  # CPUにロードする場合
        # state_dict = torch.load(model_checkpoint_path) # GPUが利用可能な場合
        model.load_state_dict(state_dict)
    except RuntimeError:
        # チェックポイントがモデル全体やオプティマイザの状態などを含む辞書の場合
        checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 上記以外の場合は、チェックポイントの内容を確認してキーを特定する必要があります。
            # 最も単純なケースとして、チェックポイントファイル自体が state_dict であると仮定してみます。
            # (最初のtryブロックでエラーになった場合、これは通常通りません)
            print("state_dict のキーが見つかりませんでした。チェックポイントの内容を確認してください。")
            # model.load_state_dict(checkpoint) # これが機能する場合もあります

    # 3. 推論モードへの切り替え
    model.eval()

    # (オプション) GPUが利用可能な場合はGPUにモデルを移動
    if torch.cuda.is_available():
        model.to('cuda')

    print("モデルの準備が完了しました。")

    # --- ここからは推論の実行例です ---

    # # 4. 画像の前処理プロセッサの準備
    # image_processor = ViTImageProcessor.from_pretrained(pretrained_model_name)

    # # 5. 推論したい画像の読み込みと前処理
    # # 例: 'path/to/your/image.jpg' を実際の画像パスに置き換えてください。
    # try:
    #     image_path = "path/to/your/image.jpg" # ★推論したい画像のパスを指定
    #     image = Image.open(image_path).convert("RGB")
    #     inputs = image_processor(images=image, return_tensors="pt")

    #     # (オプション) GPUを使用する場合、入力データもGPUに送る
    #     if torch.cuda.is_available():
    #         inputs = {k: v.to('cuda') for k, v in inputs.items()}

    #     # 6. 推論の実行
    #     with torch.no_grad(): # 勾配計算を無効化
    #         outputs = model(**inputs)
    #         logits = outputs.logits

    #     # 7. 結果の解釈
    #     predicted_class_idx = logits.argmax(-1).item()
    #     print(f"予測されたクラスID: {predicted_class_idx}")

    #     # (オプション) クラスラベルのマッピングがあれば、ラベル名を表示
    #     # label_map = {0: "cat", 1: "dog", ...} # ファインチューニング時に使用したラベルマップ
    #     # if label_map and predicted_class_idx in label_map:
    #     # print(f"予測されたラベル: {label_map[predicted_class_idx]}")

    # model = YOLO("detection_model/checkpoints/best.pt")

    # input_image = args.image

    # cards = model(input_image)
    # recognizer = HandsRecognizer(cards)
    # hand_rank, best_hand = recognizer.evaluate()

    # print(f"Best hand rank: {hand_rank.name}")
    # print("Best hand cards:")
    # for card in best_hand:
    #     print(card)
