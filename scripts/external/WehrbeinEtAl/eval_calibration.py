import torch
import numpy as np
from tqdm import tqdm

import models.model as model
import config as c
from utils.eval_functions import (
    compute_CP_list,
    pa_hypo_batch,
    err_3dpe_parallel,
    compute_3DPCK,
)
from utils.data_utils import reinsert_root_joint_torch, root_center_poses
import data.data_h36m
from sklearn.metrics import auc
from tqdm import tqdm
import wandb
import time

wandb.init(
    project="propose_human36m",
    entity="ppierzc",
    job_type="evaluation",
    name=f"wehrbein_calibration_{time.strftime('%d/%m/%Y::%H:%M:%S')}",
    config={
        "seed": 2,
    },
)

# Set seed
torch.manual_seed(wandb.config.seed)
np.random.seed(wandb.config.seed)


print("Program is running on: ", c.device)
print("EVALUATING EXPERIMENT: ", c.experiment_name, "\n")

inn = model.poseINN()
inn.to(c.device)
inn.load(c.load_model_name, c.device)
inn.eval()

print(f"{sum(p.numel() for p in inn.parameters()):,}")

c.batch_size = 512

n_hypo = 200
std_dev = 1.0

cps_min_th = 1
cps_max_th = 300
cps_step = 1
cps_length = int((cps_max_th + 1 - cps_min_th) / cps_step)

quick_eval_stride = 16
test_dataset = data.data_h36m.H36MDataset(
    c.test_file,
    quick_eval=True,
    quick_eval_stride=quick_eval_stride,
    train_set=False,
    hardsubset=False,
)

loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=True, drop_last=False
)

n_poses = len(test_dataset)

total_err_z0_p1 = 0
total_err_mean_p1 = 0
total_err_worst_p1 = 0
total_err_best_p1 = 0
total_err_occ_best_p1 = 0
total_err_median_p1 = 0

total_err_z0_p2 = 0
total_err_mean_p2 = 0
total_err_worst_p2 = 0
total_err_best_p2 = 0
total_err_median_p2 = 0

total_best_pck_oracle_p1 = 0
total_auc_cps_p2_best = torch.zeros((cps_length,))

hypo_stddev = torch.zeros((3, 17))

quantiles = np.arange(0, 1.05, 0.05)
quantile_counts = np.zeros((len(quantiles), 16))
q_val = []
total = 0

for batch_idx, sample in tqdm(enumerate(loader)):
    # if batch_idx == 10:
    #     break
    x = sample["poses_3d"]
    y_gt = sample["p2d_hrnet"]
    cond = sample["gauss_fits"]
    occl = sample["occlusions"]
    bs = x.shape[0]
    x_gt = sample["p3d_gt"]

    # sample multiple z
    z_all = std_dev * torch.randn(n_hypo, bs, c.ndim_z, device=c.device)
    y_gt = y_gt[None, :].repeat(n_hypo, 1, 1)
    y_rand = torch.cat((z_all, y_gt), dim=2)
    y_rand = y_rand.view(-1, c.ndim_y + c.ndim_z)
    cond = cond[None].repeat(n_hypo, 1, 1).view(-1, inn.cond_in_size)

    with torch.no_grad():
        poses_3d_pred = inn.reverse(y_rand, cond)

    poses_3d_pred = reinsert_root_joint_torch(poses_3d_pred)
    poses_3d_pred = root_center_poses(poses_3d_pred) * 1000
    poses_3d_pred = poses_3d_pred.view(n_hypo, 3, 17).swapaxes(1, 2)[:, 1:]

    x_gt = x_gt.view(1, 3, 17).swapaxes(1, 2)[:, 1:]

    sample_median = torch.Tensor(poses_3d_pred).median(0).values
    errors = ((sample_median - poses_3d_pred) ** 2).sum(-1) ** 0.5
    true_error = ((sample_median - x_gt) ** 2).sum(-1) ** 0.5

    q_vals = np.quantile(errors.data.numpy(), quantiles, 0)
    q_val.append(q_vals)

    v = (q_vals > true_error.data.numpy().squeeze()).astype(int)
    if not np.isnan(v).any():
        total += 1
        quantile_counts += v

quantile_freqs = quantile_counts / total

calibration_score = np.abs(np.median(quantile_freqs, axis=1) - quantiles).mean()
wandb.log({"calibration_score": calibration_score})
print("Calibration score: ", calibration_score)

np.save("wehrbein_quantile_freqs.npy", quantile_freqs)
np.save("wehrbein_quantiles.npy", quantiles)
np.save("wehrbein_q_val.npy", q_val)
