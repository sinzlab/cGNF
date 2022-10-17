import argparse
import os
import time
from pathlib import Path

import torch
import wandb
import yaml

from propose.datasets.human36m.preprocess import pickle_cameras, pickle_poses
from propose.utils.reproducibility import (
    check_uncommitted_changes,
    get_commit_hash,
    get_package_version,
)
from train.human36m import human36m

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
    "--resume",
    default="",
    type=str,
    help="Which run to resume",
)

parser.add_argument(
    "--resume_id",
    default="",
    type=str,
    help="Id of run which to resume",
)

parser.add_argument(
    "--experiment",
    default="mpii-prod.yaml",
    type=str,
    help="Experiment config file",
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
        if "cuda_accelerated" not in config:
            config["cuda_accelerated"] = torch.cuda.is_available()

        config["env"] = {
            "cgnf_version": get_commit_hash(),
            "uncommitted_changes": check_uncommitted_changes(),
            "propose_version": get_package_version("propose"),
        }

        if args.wandb:
            wandb.init(
                id=args.resume_id if args.resume_id else None,
                project="propose_human36m",
                entity=os.environ["WANDB_USER"],
                config=config,
                job_type="training",
                name=args.resume
                if args.resume
                else f"{config['experiment_name']}_{time.strftime('%d/%m/%Y::%H:%M:%S')}",
                tags=config["tags"] if "tags" in config else None,
                group=config["group"] if "group" in config else None,
                resume=bool(args.resume),
            )

        human36m(use_wandb=args.wandb, config=config)
    else:
        print(
            "Not running any scripts as no arguments were passed. Run with --help for more information."
        )
