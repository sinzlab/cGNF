import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from propose.datasets.human36m.Human36mDataset import Human36mDataset
from propose.models.flows import CondGraphFlow
from propose.models.nn.embedding import SageEmbedding
from propose.utils.reproducibility import (
    get_commit_hash,
    get_package_version,
    set_random_seed,
)


def calibration(flow, test_dataloader, occluded=True):
    print("occluded")
    total = 0
    pbar = tqdm(range(len(test_dataloader)))

    quantiles = np.arange(0, 1.05, 0.05)

    if occluded:
        quantile_counts = np.zeros((len(quantiles), 1))
    else:
        quantile_counts = np.zeros((len(quantiles), 16))
    q_val = []

    i = 0
    for batch, _, action in test_dataloader:
        i += 1
        pbar.update(1)
        batch.to(flow.device)
        samples = flow.sample(200, batch)

        true_pose = batch["x"].x.cpu().numpy().reshape(-1, 16, 1, 3)

        sample_poses = samples["x"].x.detach().cpu().numpy().reshape(-1, 16, 200, 3)

        if occluded:
            true_pose = true_pose[
                :, np.insert(action["occlusion"].bool().numpy(), 9, False)
            ]
            sample_poses = sample_poses[
                :, np.insert(action["occlusion"].bool().numpy(), 9, False)
            ]

        sample_mean = (
            torch.Tensor(sample_poses).median(-2).values.numpy()[..., np.newaxis, :]
        )
        errors = ((sample_mean / 0.0036 - sample_poses / 0.0036) ** 2).sum(-1) ** 0.5
        true_error = ((sample_mean / 0.0036 - true_pose / 0.0036) ** 2).sum(-1) ** 0.5

        q_vals = np.quantile(errors, quantiles, 2).squeeze(1)
        q_val.append(q_vals)

        if occluded:
            v = np.nanmean((q_vals > true_error.squeeze()).astype(int), axis=1)[
                :, np.newaxis
            ]
        else:
            v = (q_vals > true_error.squeeze()).astype(int)

        if not np.isnan(v).any():
            total += 1
            quantile_counts += v

        _quantile_freqs = quantile_counts / total

        # _calibration_score = np.median(_quantile_freqs, axis=1).sum() * 0.05
        _calibration_score = np.abs(
            np.median(_quantile_freqs, axis=1) - quantiles
        ).mean()

        pbar.set_description(f"Calibration score: {_calibration_score:.4f}")

    quantile_freqs = quantile_counts / total

    calibration_score = np.abs(np.median(quantile_freqs, axis=1) - quantiles).mean()
    # calibration_score = np.median(quantile_freqs, axis=1).sum() * 0.05

    print(f"{'Occluded ' if occluded else ''}Calibration score: {calibration_score}")

    return quantiles, quantile_freqs, q_val, calibration_score


def calibration_experiment(flow, config, occluded=True, **kwargs):
    test_dataset = Human36mDataset(
        **config["dataset"],
        **kwargs,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0
    )

    return calibration(flow, test_dataloader, occluded=occluded)


def run(use_wandb, config):
    set_random_seed(config["seed"])

    config["dataset"]["dirname"] = config["dataset"]["dirname"] + "/test"

    if use_wandb:
        wandb.init(
            project="propose_human36m",
            entity=os.environ["WANDB_USER"],
            config=config,
            job_type="evaluation",
            name=f"{config['experiment_name']}_calibration_{time.strftime('%d/%m/%Y::%H:%M:%S')}",
            tags=config["tags"] if "tags" in config else None,
            group=config["group"] if "group" in config else None,
        )

    flow = CondGraphFlow.from_pretrained(
        f'ppierzc/propose_human36m/{config["experiment_name"]}:latest'
    )

    config["cuda_accelerated"] = flow.set_device()
    flow.eval()

    # Test
    quantiles, quantile_freqs, q_val, calibration_score = calibration_experiment(
        flow,
        config,
        occlusion_fractions=[],
        # hardsubset=True,
        test=True,
        occluded=False,
    )

    print(quantile_freqs)

    wandb.log(
        {
            "calibration_score": calibration_score,
            "quantiles": quantiles,
            "quantile_freqs": quantile_freqs,
        }
    )

    quantiles, quantile_freqs, q_val, calibration_score = calibration_experiment(
        flow,
        config,
        occlusion_fractions=[],
        test=True,
        occluded=False,
    )

    print(quantile_freqs)

    wandb.log(
        {
            "calibration_score": calibration_score,
            "quantiles": quantiles,
            "quantile_freqs": quantile_freqs,
        }
    )

    sns.set_context("talk")
    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(5, 5), dpi=150)
        plt.fill_between(
            quantiles,
            np.mean(quantile_freqs, axis=1) + np.std(quantile_freqs, axis=1),
            np.mean(quantile_freqs, axis=1) - np.std(quantile_freqs, axis=1),
            color="#1E88E5",
            alpha=0.5,
            zorder=-5,
            rasterized=True,
        )
        plt.plot([0, 1], [0, 1], ls="--", c="tab:gray")
        plt.plot(
            quantiles,
            np.median(quantile_freqs, axis=1),
            c="#1E88E5",
            alpha=1,
            label="cGNF all",
        )
        plt.xticks(np.arange(0, 1.2, 0.2))
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.xlabel("Quantile")
        plt.ylabel("Frequency")
        plt.text(0.03, 0.07, "reference line", rotation=45, c="k", fontsize=15)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Calibration")
        plt.legend(frameon=False)

        plt.gca().set_rasterization_zorder(-1)

    if use_wandb:
        img = wandb.Image(plt)
        wandb.log({"calibration": img})

    plt.close()
