from model.model_factory import build_model
from .hands_recognition import HandsRecognizer
from argparse import ArgumentParser


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

    model = build_model("yolov11")
    input_image = args.image

    cards = model.predict(input_image)
    recognizer = HandsRecognizer(cards)
    hand_rank, best_hand = recognizer.evaluate()

    print(f"Best hand rank: {hand_rank.name}")
    print("Best hand cards:")
    for card in best_hand:
        print(card)
