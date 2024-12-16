import argparse
import sys

import tensorflow as tf

from .tools.build_dataset import build_dataset
from .tools.init_tf_env import init_tf_env


def options_parser() -> argparse.ArgumentParser:
    """
    Use to handle program parameters and options.

    Returns
    -------
    The parser object.
    """
    parser = argparse.ArgumentParser(
        prog="Distribution",
        description="This program should be used to see the distribution of plant types and states.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument("model_path", type=str, nargs=1, help="Path to the trained model.")
    parser.add_argument(
        "directory_path", type=str, nargs=1, help="Path to the directory of images."
    )
    parser.add_argument(
        "--batch_size",
        type=float,
        default=32,
        help="Used to specify how many images to process in a batch.",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.2,
        help="Used to specify the ratio of data used for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        help="Used to reproduce a dataset split.",
    )

    return parser


if __name__ == "__main__":
    try:
        init_tf_env()

        args = options_parser().parse_args()

        model_path = args.model_path[0]
        directory_path = args.directory_path[0]

        model = tf.keras.models.load_model(model_path)

        (_, evaluate_dataset) = build_dataset(
            directory_path,
            batch_size=args.batch_size,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )

        test_loss, test_acc = model.evaluate(evaluate_dataset)

        print("Evaluation dataset size", len(evaluate_dataset) * args.batch_size)
        print("Evaluation accuracy:", test_acc)

    except Exception as e:
        print(">>> Oops, something went wrong.", file=sys.stderr)
        print(e)
