from dotenv import load_dotenv

load_dotenv()

from torch_geometric.loader import DataLoader

from propose.models.flows.CondGraphFlow import CondGraphFlow

import seaborn as sns
import matplotlib.pyplot as plt

from propose.utils.reproducibility import set_random_seed

import brax
import torch
from torch_geometric.data import HeteroData
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

import wandb

set_random_seed(10)

#@title A bouncy ball scene
bouncy_ball = brax.Config(dt=0.05, substeps=4)

# ground is a frozen (immovable) infinite plane
ground = bouncy_ball.bodies.add(name='ground')
ground.frozen.all = True
plane = ground.colliders.add().plane
plane.SetInParent()  # for setting an empty oneof

# ball weighs 1kg, has equal rotational inertia along all axes, is 1m long, and
# has an initial rotation of identity (w=1,x=0,y=0,z=0) quaternion
ball = bouncy_ball.bodies.add(name='ball', mass=1)
cap = ball.colliders.add().capsule
cap.radius, cap.length = 0.05, 1

# gravity is -9.8 m/s^2 in z dimension
bouncy_ball.gravity.z = -9.8

#@title A pendulum config for Brax
pendulum = brax.Config(dt=0.01, substeps=4)

# start with a frozen anchor at the root of the pendulum
anchor = pendulum.bodies.add(name='anchor', mass=1.0)
anchor.frozen.all = True

# now add a middle and bottom ball to the pendulum
pendulum.bodies.append(ball)
pendulum.bodies.append(ball)
pendulum.bodies.append(ball)
pendulum.bodies[1].name = 'middle'
pendulum.bodies[2].name = 'middle2'
pendulum.bodies[3].name = 'bottom'

# connect anchor to middle
joint = pendulum.joints.add(name='joint1', parent='anchor',
                            child='middle', stiffness=10000, angular_damping=0)
joint.angle_limit.add(min = -180, max = 180)
joint.child_offset.z = 1
joint.rotation.z = 90

# connect middle to bottom
pendulum.joints.append(joint)
pendulum.joints[1].name = 'joint2'
pendulum.joints[1].parent = 'middle'
pendulum.joints[1].child = 'middle2'

pendulum.joints.append(joint)
pendulum.joints[2].name = 'joint3'
pendulum.joints[2].parent = 'middle2'
pendulum.joints[2].child = 'bottom'

# gravity is -9.8 m/s^2 in z dimension
pendulum.gravity.z = -9.8

def draw_system(ax, pos, alpha=1):
  for i, p in enumerate(pos):
    ax.add_patch(Circle(xy=(p[0], p[2]), radius=cap.radius, fill=True, color=(0, 0, 0, alpha)))
    if i < len(pos) - 1:
      pn = pos[i + 1]
      ax.add_line(Line2D([p[0], pn[0]], [p[2], pn[2]], color=(0, 0, 0, alpha)))


qp = brax.System(pendulum).default_qp()
qp.pos[:, 2] -= 2.5

from tqdm import tqdm
from torch.utils.data import Dataset


class PartPendulumDataset(Dataset):
    def __init__(self):
        data_list = []
        prior_data_list = []

        for i in tqdm(range(1)):
            sys = brax.System(pendulum)
            qp = sys.default_qp()
            qp.pos[:, 2] -= 2.5

            qp.vel[1, 0] = 5
            qp.vel[2, 0] = 5
            qp.vel[3, 0] = 5

            for i in range(25):
                qp, _ = sys.step(qp, [])
                qp, _ = sys.step(qp, [])

                points = torch.Tensor(qp.pos)
                vel = torch.Tensor(qp.ang)
                points[:, 1], points[:, 2] = points[:, 2].clone(), points[:, 1].clone()
                vel[:, 1], vel[:, 2] = vel[:, 2].clone(), vel[:, 1].clone()

                combs = ((0, 0), (1, 1), (2, 2))
                data = HeteroData()
                data['x'].x = points[1:, :2] + torch.randn(3, 2) * 0.1

                n_points = 3
                point_indexes = torch.arange(n_points)
                c = data['x'].x[..., :2]

                data['c'].x = c

                data['c', '->', 'x'].edge_index = torch.LongTensor([*combs]).T
                data['x', '->', 'x'].edge_index = torch.LongTensor([[0, 1], [1, 2]]).T
                data['x', '<-', 'x'].edge_index = torch.LongTensor([[0, 1], [1, 2]]).T

                data_list.append(data)

                data = HeteroData()
                data['x'].x = points[1:, :2] + torch.randn(3, 2) * 0.1

                n_points = 3
                point_indexes = torch.arange(n_points)
                c = data['x'].x[..., :2]
                v = vel[..., :2]
                data['c'].x = c

                data['x', '->', 'x'].edge_index = torch.LongTensor([[0, 1], [1, 2]]).T
                data['x', '<-', 'x'].edge_index = torch.LongTensor([[0, 1], [1, 2]]).T

                prior_data_list.append(data)

        self.data = data_list
        self.prior_data_list = prior_data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.prior_data_list[idx], {}

    def metadata(self):
        return self.data[0].metadata()


dataset = PartPendulumDataset()
data_loader = DataLoader(dataset, batch_size=100, shuffle=True)

flow = CondGraphFlow(num_layers=10, features=2, context_features=2, root_features=2)

api = wandb.Api()
artifact = api.artifact('ppierzc/pendulum_tracking/4p_pendulum:latest', type="model")

if wandb.run:
    wandb.run.use_artifact(artifact, type="model")

artifact_dir = artifact.download()

device = "cuda" if torch.cuda.is_available() else "cpu"
flow.load_state_dict(
    torch.load(artifact_dir + "/model.pt", map_location=torch.device(device)),
    strict=False,
)

flow.eval()
for batch, prior_batch, _ in data_loader:
    batch = batch.to(device)
    prior_batch = prior_batch.to(device)

    posterior_log_prob = flow.log_prob(batch)
    prior_log_prob = flow.log_prob(prior_batch)

    print(posterior_log_prob.mean().item(), prior_log_prob.mean().item())

data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
for i in range(10):
    part_data, _, _ = next(iter(data_loader))

pos = part_data['x'].x[..., :2]

flow.eval()
M_part = flow.sample(100, part_data)['x']['x'].detach().cpu()

print(torch.mean(torch.norm(pos - M_part.permute(1, 0, 2), dim=-1)))
print(M_part.std(1))
print(M_part.mean(1))
print(pos)

def set_axis_color(ax, color):
    for k, v in ax.spines.items():
        v.set_color(color)
    ax.tick_params(colors=color)

edgecolor = "k" #'.5'
tick_direction = "out"

plt.rcParams["axes.axisbelow"] = False

sns.set_context('talk')
fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)

ax.scatter([0], [1], s=100, c='k', marker='o')
colors = [
    (0, 0, 0, 1), # blue
    (0, 0, 0, 1), # green
    (0, 0, 0, 1) # red
]
for j in [0, 1, 2]:
    xy = pos[j].numpy()
    ax.scatter(xy[0], xy[1], s=100, c='k', marker='o')

# plot line through the points in pos
plt.plot([0, pos[0, 0].item(), pos[1, 0].item(), pos[2, 0].item()], [1, pos[0, 1].item(), pos[1, 1].item(), pos[2, 1].item()], color=(0, 0, 0, 1), linewidth=3)

plt.xlim(-1, 1)
plt.ylim(-3, 2)

ax.scatter(M_part[0, :, 0], M_part[0, :, 1], c='#EDAC32', alpha=1, edgecolor='none', s=15, zorder=-1)
ax.scatter(M_part[1, :, 0], M_part[1, :, 1], c='#D81B60', alpha=1, edgecolor='none', s=15, zorder=-1)
ax.scatter(M_part[2, :, 0], M_part[2, :, 1], c='#1E88E5', alpha=1, edgecolor='none', s=15, zorder=-1)

set_axis_color(ax, edgecolor)
ax.tick_params(direction=tick_direction, labelsize=20)

ax.set_title("cGNF zero-shot", fontsize=20)

plt.bar([-10], [0], color="#EDAC32", edgecolor='k', label='$x_1$', linewidth=3)
plt.bar([-10], [0], color="#D81B60", edgecolor='k', label='$x_2$', linewidth=3)
plt.bar([-10], [0], color="#1E88E5", edgecolor='k', label='$x_3$', linewidth=3)
plt.legend(loc="upper right", frameon=False, handletextpad=0.4, fontsize=22)
sns.despine()
plt.savefig('./pendulum_cgnf.png', bbox_inches='tight')
plt.savefig('./pendulum_cgnf.pdf', bbox_inches='tight')

plt.show()
