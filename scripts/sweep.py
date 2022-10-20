import argparse
import os
from functools import partial
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
from sweep.human36m import human36m

parser = argparse.ArgumentParser(description="Arguments for running the scripts")

parser.add_argument(
    "--human36m",
    default=False,
    action="store_true",
    help="Run the training script for the Human 3.6m dataset",
)

parser.add_argument(
    "--wandb",
    default=True,
    action="store_true",
    help="Whether to use wandb for logging (required for sweeping)",
)

parser.add_argument(
    "--sweep",
    default="sweep",
    type=str,
    help="Sweep config file",
)

parser.add_argument(
    "--sweep_id",
    type=str,
    help="Sweep ID to use if sweep is already running",
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

    if not args.wandb:
        raise ValueError("Wandb is required for sweeping.")

    dataset = Path("")
    if args.human36m:
        dataset = Path("human36m")

    config_file = Path(args.sweep + ".yaml")
    config_file = Path("/sweeps") / dataset / config_file

    train_config_file = (
        Path("/sweeps") / dataset / Path(args.sweep + "_train_config.yaml")
    )

    with open(config_file, "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)

    with open(train_config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if "name" in sweep_config:
            config["experiment_name"] = sweep_config["name"]

    if args.human36m:
        if "cuda_accelerated" not in config:
            config["cuda_accelerated"] = torch.cuda.is_available()

        sweep_id = args.sweep_id
        if not sweep_id:
            sweep_id = wandb.sweep(
                sweep_config,
                project="propose_human36m",
                entity=os.environ["WANDB_USER"],
            )

        config["env"] = {
            "cgnf_version": get_commit_hash(),
            "uncommitted_changes": check_uncommitted_changes(),
            "propose_version": get_package_version("propose"),
        }

        run_func = partial(human36m, use_wandb=args.wandb, config=config)

        wandb.agent(
            sweep_id,
            function=run_func,
            count=config["sweep"]["count"],
            project="propose_human36m",
            entity=os.environ["WANDB_USER"],
        )
    else:
        print(
            "Not running any scripts as no arguments were passed. Run with --help for more information."
        )
