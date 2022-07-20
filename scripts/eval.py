from propose.utils.imports import dynamic_import

import argparse

import os
import yaml

from pathlib import Path


parser = argparse.ArgumentParser(description="Arguments for running the scripts")

parser.add_argument(
    "--human36m",
    default=False,
    action="store_true",
    help="Run the training script for the Human 3.6m dataset",
)

parser.add_argument(
    "--wandb",
    default=False,
    action="store_true",
    help="Whether to use wandb for logging",
)

parser.add_argument(
    "--experiment",
    default="mpii-prod.yaml",
    type=str,
    help="Experiment config file",
)

parser.add_argument(
    "--script",
    default="eval.human36m.human36m",
    type=str,
    help="Experiment script",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.wandb:
        if not os.environ["WANDB_API_KEY"]:
            raise ValueError(
                "Wandb API key not set. Please set the WANDB_API_KEY environment variable."
            )
        if not os.environ["WANDB_USER"]:
            raise ValueError(
                "Wandb user not set. Please set the WANDB_USER environment variable."
            )

    dataset = Path("")
    if args.human36m:
        dataset = Path("human36m")

    config_file = Path(args.experiment + ".yaml")
    config_file = Path("/experiments") / dataset / config_file

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        if "experiment_name" not in config:
            config["experiment_name"] = args.experiment

    if args.human36m:
        dynamic_import(args.script, "run")(use_wandb=args.wandb, config=config)
    else:
        print(
            "Not running any scripts as no arguments were passed. Run with --help for more information."
        )
