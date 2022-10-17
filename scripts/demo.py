import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from neuralpredictors.data.transforms import rescale
from torchvision.transforms import Pad

from propose.models.detectors import HRNet
from propose.models.flows import CondGraphFlow
from propose.poses.human36m import Human36mPose, MPIIPose

# Get models
flow = CondGraphFlow.from_pretrained("ppierzc/propose_human36m/mpii-prod-xlarge:v20")
# model = HRNet.from_pretrained("ppierzc/cgnf/hrnet:v0")
from propose.models.detectors.hrnet.config import config

config_file = "../data/models/w32_256x256_adam_lr1e-3.yaml"

config.defrost()
config.merge_from_file(config_file)
config.freeze()

model = HRNet(config)

from collections import OrderedDict

state_dict = torch.load(
    "../data/models" + "/fine_HRNet.pt",
    map_location=torch.device("cpu"),
)["net"]

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k  # remove module.
    new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=False)
yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

flow.eval()
model.eval()
yolo.eval()


def process_samples(samples):
    samples = samples.detach().cpu().numpy()
    samples = np.insert(samples, 0, 0, 0)
    return samples.swapaxes(0, 1)


def d2_pose_estimation(input_image):
    with torch.no_grad():
        pred_image = model.preprocess(input_image[np.newaxis])
        coords, vals = model.pose_estimate(pred_image)
        pose_2d = MPIIPose(coords)
        pose_2d.occluded_markers = vals < 0.3
        pose_2d = pose_2d.to_human36m()

        context_graph = flow.preprocess(pose_2d)
        samples = flow.sample(100, context=context_graph)["x"]["x"]
        samples = process_samples(samples)

    mode_sample = np.median(samples, axis=0)

    sampled_pose = Human36mPose(samples)
    mode_pose = Human36mPose(mode_sample)

    fig = plt.figure(figsize=(30, 10))
    ax = plt.subplot(1, 3, 1, projection="3d")
    ax.view_init(0, 90)
    sampled_pose.plot(ax, plot_type="None", c="k", alpha=0.05)
    mode_pose.plot(ax, plot_type="None", c="r", alpha=1)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)

    ax = plt.subplot(1, 3, 2, projection="3d")
    ax.view_init(0, 60)
    sampled_pose.plot(ax, plot_type="None", c="k", alpha=0.05)
    mode_pose.plot(ax, plot_type="None", c="r", alpha=1)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)

    ax = plt.subplot(1, 3, 3, projection="3d")
    ax.view_init(0, 30)
    sampled_pose.plot(ax, plot_type="None", c="k", alpha=0.05)
    mode_pose.plot(ax, plot_type="None", c="r", alpha=1)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    plt.close()

    return fig


demo = gr.Interface(d2_pose_estimation, gr.Image(), ["plot"])


demo.launch(server_name="0.0.0.0")
