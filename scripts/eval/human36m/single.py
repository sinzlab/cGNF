from propose.datasets.human36m.Human36mDataset import Human36mDataset
from torch_geometric.loader import DataLoader

from propose.utils.reproducibility import set_random_seed
from propose.evaluation.mpjpe import mpjpe, pa_mpjpe
from propose.evaluation.pck import pck, human36m_joints_to_use

from propose.models.flows import CondGraphFlow

import os

import time
from tqdm import tqdm
import numpy as np

import wandb


def evaluate(flow, test_dataloader, temperature=1.0):
    single_mpjpes = []
    single_pa_mpjpes = []
    pck_scores = []
    mean_pck_scores = []

    iter_dataloader = iter(test_dataloader)

    pbar = tqdm(range(len(test_dataloader)))

    for _ in pbar:
        batch, _, action = next(iter_dataloader)
        batch.to(flow.device)

        samples = flow.mode_sample(batch)

        true_pose = batch["x"].x.cpu().numpy().reshape(-1, 16, 1, 3)
        sample_poses = samples["x"].x.detach().cpu().numpy().reshape(-1, 16, 1, 3)

        true_pose = np.insert(true_pose, 0, 0, axis=1)
        sample_poses = np.insert(sample_poses, 0, 0, axis=1)

        pck_score = pck(
            true_pose[:, human36m_joints_to_use] / 0.0036,
            sample_poses[:, human36m_joints_to_use] / 0.0036,
        )

        has_correct_pose = pck_score.max().unsqueeze(0).numpy()
        mean_correct_pose = pck_score.mean().unsqueeze(0).numpy()

        m = mpjpe(true_pose / 0.0036, sample_poses / 0.0036, dim=1)
        m_single = m[..., 0]

        pa_m = (
            pa_mpjpe(true_pose[0] / 0.0036, sample_poses[0] / 0.0036, dim=0)
            .unsqueeze(0)
            .numpy()
        )

        pa_m_single = pa_m[..., 0]

        m_single = m_single.tolist()

        single_mpjpes += [m_single]
        single_pa_mpjpes += [pa_m_single]

        pck_scores += [has_correct_pose]
        mean_pck_scores += [mean_correct_pose]

        pbar.set_description(
            f"Single MPJPE: {np.concatenate(single_mpjpes).mean():.4f} "
            f"Single PA MPJPE: {np.concatenate(single_pa_mpjpes).mean():.4f} "
            f"PCK: {np.concatenate(pck_scores).mean():.4f} "
            f"Mean PCK: {np.concatenate(mean_pck_scores).mean():.4f} "
        )

    return (
        single_mpjpes,
        single_pa_mpjpes,
        pck_scores,
        mean_pck_scores,
    )


def mpjpe_experiment(flow, config, name="test", **kwargs):
    test_dataset = Human36mDataset(
        **config["dataset"],
        **kwargs,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, pin_memory=False, num_workers=0
    )
    (
        test_res_single,
        test_res_pa_single,
        test_res_pck,
        test_res_mean_pck,
    ) = evaluate(flow, test_dataloader)

    res = {
        f"{name}/test_res_single": np.concatenate(test_res_single).mean(),
        f"{name}/test_res_pa_single": np.concatenate(test_res_pa_single).mean(),
        f"{name}/test_res_pck": np.concatenate(test_res_pck).mean(),
        f"{name}/test_res_mean_pck": np.concatenate(test_res_mean_pck).mean(),
    }

    return res, test_dataset, test_dataloader


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
            name=f"{config['experiment_name']}_single_{time.strftime('%d/%m/%Y::%H:%M:%S')}",
            tags=config["tags"] if "tags" in config else None,
            group=config["group"] if "group" in config else None,
        )

    flow = CondGraphFlow.from_pretrained(
        f'ppierzc/propose_human36m/{config["experiment_name"]}:v20'
    )

    config["cuda_accelerated"] = flow.set_device()
    flow.eval()

    # Test
    test_res, test_dataset, test_dataloader = mpjpe_experiment(
        flow,
        config,
        occlusion_fractions=[],
        test=True,
        name="test",
    )

    if use_wandb:
        wandb.log(test_res)

    # Hard
    hard_res, hard_dataset, hard_dataloader = mpjpe_experiment(
        flow,
        config,
        occlusion_fractions=[],
        hardsubset=True,
        name="hard",
    )

    if use_wandb:
        wandb.log(hard_res)

    hard_dataset = Human36mDataset(
        **config["dataset"],
        occlusion_fractions=[],
        hardsubset=True,
    )

    # Occlusion Only
    mpjpes = []
    for i in tqdm(range(len(hard_dataset))):
        batch = hard_dataset[i][0]
        batch.cuda()
        samples = flow.mode_sample(batch.cuda())

        true_pose = (
            batch["x"]
            .x.cpu()
            .numpy()
            .reshape(-1, 16, 1, 3)[:, np.insert(hard_dataset.occlusions[i], 9, False)]
        )
        sample_poses = (
            samples["x"]
            .x.detach()
            .cpu()
            .numpy()
            .reshape(-1, 16, 1, 3)[:, np.insert(hard_dataset.occlusions[i], 9, False)]
        )

        m = mpjpe(true_pose / 0.0036, sample_poses / 0.0036, dim=1)
        m = np.min(m, axis=-1)

        m = m.tolist()

        mpjpes += [m]

    occl_res = np.nanmean(mpjpes)
    if use_wandb:
        wandb.log({"occl/best_mpjpe": occl_res})

    print("MPJPE for best")
    print("---")
    # print(f"H36M: {test_res}")
    # print(f"H36MA: {hard_res}")
    print(f"Occl.: {occl_res}")
    print("---")
