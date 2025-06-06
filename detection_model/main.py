import argparse

from detection_model.model_factory import build_model
from detection_model.train import train_model
from detection_model.val import validate_model


def parse_option():
    parser = argparse.ArgumentParser(
        'Model training and evaluation script', add_help=False)
    # easy config modification
    parser.add_argument('--batch_size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--epochs', type=int, default=300,
                        help="number of epochs to train")
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--output', type=str, default='./checkpoints',
                        help='path to save the trained model')
    parser.add_argument('--model', type=str, default='yolov11',
                        help='model name to use for training')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate the model after training')
    args = parser.parse_args()
    return args


def main():
    args = parse_option()

    # Build the model
    model = build_model(args.model)

    # Train the model
    model = train_model(model, args)

    # Evaluate the model if specified
    if args.eval:
        validate_model(model, args)


if __name__ == '__main__':
    main()
