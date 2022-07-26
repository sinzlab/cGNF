from propose.models.detectors import HRNet
from propose.models.flows import CondGraphFlow

from propose.poses.human36m import MPIIPose, Human36mPose

import numpy as np

import gradio as gr

import torch

import matplotlib.pyplot as plt

from neuralpredictors.data.transforms import rescale
from torchvision.transforms import Pad


# Get models
flow = CondGraphFlow.from_pretrained("ppierzc/propose_human36m/mpii-prod-xlarge:v20")
model = HRNet.from_pretrained("ppierzc/cgnf/hrnet:v0")
yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

flow.eval()
model.eval()
yolo.eval()


def crop_image_to_human(input_image):
    detections = yolo(input_image)
    detections = (
        detections.pandas()
        .xyxy[0][detections.pandas().xyxy[0].name == "person"]
        .reset_index()
    )
    bbox = detections.iloc[0]

    xy = (bbox["xmin"], bbox["ymax"])
    width = bbox["xmax"] - bbox["xmin"]
    height = bbox["ymax"] - bbox["ymin"]

    center = (xy[0] + width / 2, xy[1] - height / 2)

    side = max([width, height])
    # side = min([longer, input_image.shape[0], input_image.shape[1]])

    crop_size = [
        int(center[0] - side / 2),
        int(center[0] + side / 2),
        int(center[1] - side / 2),
        int(center[1] + side / 2),
    ]
    for i in range(4):
        crop_size[i] = max([crop_size[i], 0])

    cropped_image = input_image[
        crop_size[2] : crop_size[3], crop_size[0] : crop_size[1]
    ]

    padder = Pad(
        (
            int((max(cropped_image.shape) - cropped_image.shape[0]) / 2),
            int((max(cropped_image.shape) - cropped_image.shape[1]) / 2),
        )
    )
    cropped_image = padder(torch.Tensor(cropped_image)).numpy()
    cropped_image = cropped_image / 255

    cropped_image = rescale(
        cropped_image, 256 / cropped_image.shape[0], multichannel=True
    )
    cropped_image = cropped_image[:256, :256]
    padder = Pad(
        (
            256 - cropped_image.shape[0],
            256 - cropped_image.shape[1],
        )
    )

    cropped_image = padder(torch.Tensor(cropped_image)).numpy()
    cropped_image = cropped_image[:256, :256]

    return cropped_image


def process_coords(coords, vals):
    pose_2d = MPIIPose(coords * 0.0139 * 0.8)
    pose_2d.occluded_markers = get_occlusion_vector(vals)
    pose_2d = pose_2d.to_human36m()
    pose_2d.pose_matrix = pose_2d.pose_matrix - pose_2d.pose_matrix[:, 0]
    pose_2d.pose_matrix[..., 1] = -pose_2d.pose_matrix[..., 1]

    return pose_2d


def process_samples(samples):
    samples = samples.detach().cpu().numpy()
    samples = np.insert(samples, 0, 0, 0)
    return samples.swapaxes(0, 1)


def get_occlusion_vector(vals, threshold=0.3):
    return vals < threshold


def d2_pose_estimation(input_image):
    cropped_image = crop_image_to_human(input_image)

    pred_image = torch.Tensor(cropped_image)
    pred_image = pred_image.view(1, *pred_image.shape)
    pred_image = pred_image.permute(0, 3, 1, 2)

    with torch.no_grad():
        coords, vals = model.pose_estimate(pred_image)

    pose_2d = process_coords(coords, vals)

    pose_3d = Human36mPose(np.zeros((1, 17, 3)))

    context_graph = pose_3d.conditional_graph(pose_2d)

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
