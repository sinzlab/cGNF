from pathlib import Path
from propose.datasets.human36m.preprocess import pickle_poses, pickle_cameras

import argparse

parser = argparse.ArgumentParser(description="Arguments for running the scripts")

parser.add_argument(
    "--human36m",
    default=False,
    action="store_true",
    help="Run the preprocess script for the Human 3.6m dataset",
)

parser.add_argument(
    "--rat7m",
    default=False,
    action="store_true",
    help="Run the preprocess script for the Rat 7m dataset",
)

parser.add_argument(
    "--test",
    default=False,
    action="store_true",
    help="Whether the test dataset should be processed",
)

parser.add_argument(
    "--universal",
    default=False,
    action="store_true",
    help="Whether the universal dataset should be processed",
)


def human36m(test=False, universal=False):
    input_dir = Path("/data/human36m/test/") if test else Path("/data/human36m/raw/")
    output_dir = (
        Path("/data/human36m/processed/test/")
        if test
        else Path("/data/human36m/processed/")
    )

    print(" 🥒 Pickling Human3.6M cameras")
    pickle_cameras(input_dir, output_dir)
    print(" 🥒 Pickling Human3.6M poses")
    pickle_poses(input_dir, output_dir, test=test, universal=universal)
    print("Done! 🎉")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.human36m:
        human36m(args.test)

    if args.rat7m:
        raise NotImplementedError(
            "Rat7m data preprocessing is not yet implemented. Look at the notebook preprocess_rat7m.ipynb for more information."
        )

    if not args.human36m and not args.rat7m:
        raise ValueError(
            "No dataset specified. Please use --human36m or --rat7m to specify a dataset to preprocess."
        )
