import argparse

from .model_factory import build_model
from .train import train_model
from .val import validate_model


def parse_option():
    parser = argparse.ArgumentParser(
        'Model training and evaluation script', add_help=False)
    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--epochs', type=int, default=300,
                        help="number of epochs to train")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--output', type=str, default='./checkpoints',
                        help='path to save the trained model')
    parser.add_argument('--model', type=str, default='yolov11',
                        help='model name to use for training')
    args = parser.parse_args()
    return args


def main():
    args = parse_option()

    # Build the model
    model = build_model(args.model)

    # Train the model
    model = train_model(model, args)

    # Validate the model
    # validate_model(model, args)


if __name__ == '__main__':
    main()
