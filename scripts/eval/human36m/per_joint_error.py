from propose.datasets.human36m.Human36mDataset import Human36mDataset
from torch_geometric.loader import DataLoader
from propose.poses.human36m import Human36mPose

from propose.utils.reproducibility import set_random_seed
from propose.evaluation.mpjpe import mpjpe

from propose.models.flows import CondGraphFlow

import os

import time
from tqdm import tqdm
import numpy as np

import wandb

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate(flow, test_dataloader, temperature=1.0):
    mpjpes_not_occuled = []
    mpjpes_occuled = []

    iter_dataloader = iter(test_dataloader)
    for _ in tqdm(range(len(test_dataloader))):
        batch, _, action = next(iter_dataloader)
        occluded_joints = action["occlusion"].bool().numpy()

        batch = batch.to(flow.device)
        samples = flow.sample(200, batch, temperature=temperature)

        true_pose = batch["x"].x.cpu().numpy().reshape(-1, 16, 1, 3)
        sample_poses = samples["x"].x.detach().cpu().numpy().reshape(-1, 16, 200, 3)

        true_pose = np.insert(true_pose, 0, 0, axis=1)
        sample_poses = np.insert(sample_poses, 0, 0, axis=1)

        m = mpjpe(true_pose / 0.0036, sample_poses / 0.0036, mean=False)
        m = np.min(m, axis=-1)

        m = np.delete(m, 0, axis=1)
        m = np.delete(m, 8, axis=1)

        # if occluded add values to mpjpes_occuled with the unoclluded as nan
        m_occlued = m.copy()
        m_occlued[~occluded_joints] = np.nan
        mpjpes_occuled.append(m_occlued)

        # if not occluded add values to mpjpes_not_occuled with the occluded as nan
        m_not_occlued = m.copy()
        m_not_occlued[occluded_joints] = np.nan
        mpjpes_not_occuled.append(m_not_occlued)

    return mpjpes_not_occuled, mpjpes_occuled


def mpjpe_experiment(flow, config, **kwargs):
    test_dataset = Human36mDataset(**config["dataset"], **kwargs)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, pin_memory=False, num_workers=0
    )
    mpjpes_not_occuled, mpjpes_occuled = evaluate(flow, test_dataloader)

    return np.concatenate(mpjpes_not_occuled).T, np.concatenate(mpjpes_occuled).T


def run(use_wandb: bool = False, config: dict = None):
    """
    Train a CondGraphFlow on the Human36m dataset.
    :param use_wandb: Whether to use wandb for logging.
    :param config: A dictionary of configuration parameters.
    """
    set_random_seed(config["seed"])

    config["dataset"]["dirname"] = config["dataset"]["dirname"] + "/test"

    if use_wandb:
        wandb.init(
            project="propose_human36m",
            entity=os.environ["WANDB_USER"],
            config=config,
            job_type="evaluation",
            name=f"{config['experiment_name']}_pje_{time.strftime('%d/%m/%Y::%H:%M:%S')}",
            tags=config["tags"] if "tags" in config else None,
            group=config["group"] if "group" in config else None,
        )

    flow = CondGraphFlow.from_pretrained(
        f'ppierzc/propose_human36m/{config["experiment_name"]}:latest'
    )

    config["cuda_accelerated"] = flow.set_device()
    flow.eval()

    pose = Human36mPose(np.zeros((16, 2)))
    marker_names = pose.marker_names[1:]
    del marker_names[8]

    # Test
    mpjpes_not_occuled, mpjpes_occuled = mpjpe_experiment(
        flow,
        config,
        occlusion_fractions=[],
        test=True,
    )

    df_occluded = pd.DataFrame(
        {key: value for key, value in zip(marker_names, mpjpes_occuled)}
    )

    df_not_occluded = pd.DataFrame(
        {key: value for key, value in zip(marker_names, mpjpes_not_occuled)}
    )

    df = (
        pd.concat(
            [df_not_occluded, df_occluded], keys=["not_occluded", "occluded"], axis=1
        )
        .stack()
        .stack()
        .to_frame()
        .reset_index()
    )

    plt.figure(figsize=(15, 5))
    sns.barplot(data=df, x="level_1", y=0, hue="level_2")
    plt.xticks(rotation=90)
    plt.ylabel("MPJPE")
    plt.xlabel("Joint")
    plt.legend(title="Occluded?")
    plt.tight_layout()

    output = {
        "img": wandb.Image(plt.gcf(), caption="MPJPE"),
        "occluded": {
            key: list(filter(lambda x: x, value))
            for key, value in zip(marker_names, mpjpes_occuled)
        },
        "not_occluded": {
            key: list(filter(lambda x: x, value))
            for key, value in zip(marker_names, mpjpes_not_occuled)
        },
    }

    if use_wandb:
        wandb.log(output)

    plt.close()
