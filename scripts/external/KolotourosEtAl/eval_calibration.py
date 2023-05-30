"""
Script used for evaluating the 3D pose errors of the Probabilistic Skeleton model (mode + minimum) on Human3.6M.

Example usage:
python eval_skeleton.py --checkpoint=/path/to/checkpoint

Running the above will compute the Reconstruction Error for the mode as well as the minimum error for the test set of 3DPW.
"""
import torch
import argparse
from tqdm import tqdm
from prohmr.configs import get_config, proskeleton_config, dataset_config
from prohmr.models import ProSkeleton
from prohmr.utils import Evaluator, recursive_to
from prohmr.datasets import SkeletonDataset

parser = argparse.ArgumentParser(description='Evaluate trained models')
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
parser.add_argument('--num_samples', type=int, default=200, help='Number of test samples to draw')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
parser.add_argument('--log_freq', type=int, default=10, help='How often to log results')
parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')

args = parser.parse_args()

# Use the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model config
if args.model_cfg is None:
    model_cfg = proskeleton_config()
else:
    model_cfg = get_config(args.model_cfg)

# Load dataset config
dataset_cfg = dataset_config()['H36M-VAL-P2']

# Update number of test samples drawn to the desired value
model_cfg.defrost()
model_cfg.TRAIN.NUM_TEST_SAMPLES = args.num_samples
model_cfg.freeze()

# Setup model
model = ProSkeleton.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
model.eval()

# Create dataset and data loader
mean_stats = dataset_config()['H36M-TRAIN'].MEAN_STATS
dataset = SkeletonDataset(model_cfg, dataset_cfg.DATASET_FILE, mean_stats, train=False)
dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

# List of metrics to log
metrics = ['mode_mpjpe', 'min_mpjpe', 'mode_re', 'min_re']

# Setup evaluator object
evaluator = Evaluator(dataset_length=len(dataset), keypoint_list=list(range(17)), pelvis_ind=0, metrics=metrics)

# Go over the images in the dataset.

import numpy as np

quantiles = np.arange(0, 1.05, 0.05)
quantile_counts = np.zeros((len(quantiles), 16))
total = 0
pbar = tqdm(dataloader)
for i, batch in enumerate(pbar):
    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)
    pred_keypoints_3d = out['pred_keypoints_3d'].cpu().numpy()
    gt_keypoints_3d = batch['keypoints_3d'][..., :-1].cpu()

    sample_mean = (
        torch.Tensor(pred_keypoints_3d).median(1).values.numpy()[:, np.newaxis]
    )
    errors = ((sample_mean - pred_keypoints_3d) ** 2).sum(-1) ** 0.5

    true_error = ((sample_mean - gt_keypoints_3d.unsqueeze(1).numpy()) ** 2).sum(-1) ** 0.5

    q_vals = np.quantile(errors, quantiles, 1)

    # print(q_vals.shape, true_error.shape)

    v = (q_vals > true_error.squeeze()).astype(int).sum(1)

    quantile_counts += v
    total += gt_keypoints_3d.shape[0]

    _quantile_freqs = quantile_counts / total

    quantile_freqs = np.median(_quantile_freqs, axis=1)
    ece = np.abs(quantile_freqs - quantiles).mean()

    pbar.set_description(f"ECE: {ece:.4f}")


