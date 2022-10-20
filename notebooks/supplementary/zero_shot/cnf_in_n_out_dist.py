import brax
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from propose.models.flows.Flow import NF
from propose.utils.reproducibility import set_random_seed

set_random_seed(1)

# @title A bouncy ball scene
bouncy_ball = brax.Config(dt=0.05, substeps=4)

# ground is a frozen (immovable) infinite plane
ground = bouncy_ball.bodies.add(name="ground")
ground.frozen.all = True
plane = ground.colliders.add().plane
plane.SetInParent()  # for setting an empty oneof

# ball weighs 1kg, has equal rotational inertia along all axes, is 1m long, and
# has an initial rotation of identity (w=1,x=0,y=0,z=0) quaternion
ball = bouncy_ball.bodies.add(name="ball", mass=1)
cap = ball.colliders.add().capsule
cap.radius, cap.length = 0.05, 1

# gravity is -9.8 m/s^2 in z dimension
bouncy_ball.gravity.z = -9.8

# @title A pendulum config for Brax
pendulum = brax.Config(dt=0.01, substeps=4)

# start with a frozen anchor at the root of the pendulum
anchor = pendulum.bodies.add(name="anchor", mass=1.0)
anchor.frozen.all = True

# now add a middle and bottom ball to the pendulum
pendulum.bodies.append(ball)
pendulum.bodies.append(ball)
pendulum.bodies.append(ball)
pendulum.bodies[1].name = "middle"
pendulum.bodies[2].name = "middle2"
pendulum.bodies[3].name = "bottom"

# connect anchor to middle
joint = pendulum.joints.add(
    name="joint1", parent="anchor", child="middle", stiffness=10000, angular_damping=0
)
joint.angle_limit.add(min=-180, max=180)
joint.child_offset.z = 1
joint.rotation.z = 90

# connect middle to bottom
pendulum.joints.append(joint)
pendulum.joints[1].name = "joint2"
pendulum.joints[1].parent = "middle"
pendulum.joints[1].child = "middle2"

pendulum.joints.append(joint)
pendulum.joints[2].name = "joint3"
pendulum.joints[2].parent = "middle2"
pendulum.joints[2].child = "bottom"

# gravity is -9.8 m/s^2 in z dimension
pendulum.gravity.z = -9.8


def draw_system(ax, pos, alpha=1):
    for i, p in enumerate(pos):
        ax.add_patch(
            Circle(
                xy=(p[0], p[2]), radius=cap.radius, fill=True, color=(0, 0, 0, alpha)
            )
        )
        if i < len(pos) - 1:
            pn = pos[i + 1]
            ax.add_line(Line2D([p[0], pn[0]], [p[2], pn[2]], color=(0, 0, 0, alpha)))


qp = brax.System(pendulum).default_qp()
qp.pos[:, 2] -= 2.5


class PendulumDataset(Dataset):
    def __init__(self, n=10, velocity_random=True):
        xs = []
        cs = []

        for i in tqdm(range(n)):
            sys = brax.System(pendulum)
            qp = sys.default_qp()
            qp.pos[:, 2] -= 2.5
            if velocity_random:
                qp.vel[1, 0] = torch.randn(1) * 10
                qp.vel[2, 0] = torch.randn(1) * 10
                qp.vel[3, 0] = torch.randn(1) * 10
            else:
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

                points = points[1:, :2]

                c = points[[0, 1, 2], :2]  # + torch.randn(3, 2) * 0.1

                xs.append(points + torch.randn(3, 2) * 0.05)
                cs.append(c)

        self.xs = xs
        self.cs = cs

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.cs[idx]


dataset = PendulumDataset(n=50)
data_loader = DataLoader(dataset, batch_size=100, shuffle=True)

epochs = 100
lr = 0.001
weight_decay = 0  # 1e-5

flow = NF(2 * 3, context_features=2 * 3, num_layers=20, hidden_features=(100, 200, 100))
optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3, threshold=5e-2, verbose=True
)

for epoch in tqdm(range(epochs)):
    l = []
    for x, c in data_loader:
        optimizer.zero_grad()
        c[:, -1] = 0
        c[:, -2] = 0
        loss = -flow.log_prob(x.reshape(-1, 2 * 3), c.reshape(-1, 2 * 3))
        loss = loss.mean()

        l.append(loss)

        loss.backward()
        optimizer.step()

    lr_scheduler.step(torch.mean(torch.stack(l)))

test_dataset = PendulumDataset(n=1, velocity_random=False)
data_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)
flow.eval()


_x = torch.Tensor([[[0.0031, -0.1020], [0.0579, -1.1243], [0.1886, -2.0365]]])
_c = _x[..., :2].clone()

_c[:, -1] = 0
ll = flow.log_prob(_x.reshape(-1, 2 * 3), _c.reshape(-1, 2 * 3))
ll = ll.mean()

pos = _x[0]

print("Eval of plot", ll.item())
# c[:, -1] = 0
M_part = flow.sample(100, _c.reshape(-1, 2 * 3)).detach().numpy()
M_part = M_part.reshape(-1, 3, 2)

########################################################################################################################


def set_axis_color(ax, color):
    for k, v in ax.spines.items():
        v.set_color(color)
    ax.tick_params(colors=color)


edgecolor = "k"  #'.5'
tick_direction = "out"

plt.rcParams["axes.axisbelow"] = False

sns.set_context("talk")
fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)

ax.scatter([0], [1], s=100, c="k", marker="o")
colors = [(0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1)]  # blue  # green  # red
for j in [0, 1, 2]:
    xy = pos[j].numpy()
    ax.scatter(xy[0], xy[1], s=100, c="k", marker="o")

# plot line through the points in pos
plt.plot(
    [0, pos[0, 0].item(), pos[1, 0].item(), pos[2, 0].item()],
    [1, pos[0, 1].item(), pos[1, 1].item(), pos[2, 1].item()],
    color=(0, 0, 0, 1),
    linewidth=3,
)

plt.xlim(-1, 1)
plt.ylim(-3, 2)

ax.scatter(
    M_part[:, 0, 0],
    M_part[:, 0, 1],
    c="#EDAC32",
    alpha=1,
    edgecolor="none",
    s=15,
    zorder=-1,
)
ax.scatter(
    M_part[:, 1, 0],
    M_part[:, 1, 1],
    c="#D81B60",
    alpha=1,
    edgecolor="none",
    s=15,
    zorder=-1,
)
ax.scatter(
    M_part[:, 2, 0],
    M_part[:, 2, 1],
    c="#1E88E5",
    alpha=1,
    edgecolor="none",
    s=15,
    zorder=-1,
)

set_axis_color(ax, edgecolor)
ax.tick_params(direction=tick_direction, labelsize=20)

ax.set_title("CNF out of distribution", fontsize=20)
sns.despine()
plt.savefig("./pendulum_cnf_out.png", bbox_inches="tight")
plt.savefig("./pendulum_cnf_out.pdf", bbox_inches="tight")
plt.show()


########################################################################################################################

flow = NF(2 * 3, context_features=2 * 3, num_layers=20, hidden_features=(100, 200, 100))
optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3, threshold=5e-2, verbose=True
)

data_loader = DataLoader(dataset, batch_size=100, shuffle=True)

for epoch in tqdm(range(epochs)):
    l = []
    for x, c in data_loader:
        optimizer.zero_grad()
        loss = -flow.log_prob(x.reshape(-1, 2 * 3), c.reshape(-1, 2 * 3))
        loss = loss.mean()

        l.append(loss)

        loss.backward()
        optimizer.step()

    lr_scheduler.step(torch.mean(torch.stack(l)))


data_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)
flow.eval()

for x, c in data_loader:
    loss = flow.log_prob(x.reshape(-1, 2 * 3), c.reshape(-1, 2 * 3))
    loss = loss.mean()

    print("Eval", loss.item())

flow.eval()

_x = torch.Tensor([[[0.0031, -0.1020], [0.0579, -1.1243], [0.1886, -2.0365]]])
_c = _x[..., :2]


ll = flow.log_prob(_x.reshape(-1, 2 * 3), _c.reshape(-1, 2 * 3))
ll = ll.mean()

pos = _x[0]

print("Eval of plot", ll.item())
# c[:, -1] = 0
M_part = flow.sample(100, _c.reshape(-1, 2 * 3)).detach().numpy()
M_part = M_part.reshape(-1, 3, 2)

########################################################################################################################


def set_axis_color(ax, color):
    for k, v in ax.spines.items():
        v.set_color(color)
    ax.tick_params(colors=color)


edgecolor = "k"  #'.5'
tick_direction = "out"

plt.rcParams["axes.axisbelow"] = False

sns.set_context("talk")
fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)

ax.scatter([0], [1], s=100, c="k", marker="o")
colors = [(0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1)]  # blue  # green  # red
for j in [0, 1, 2]:
    xy = pos[j].numpy()
    ax.scatter(xy[0], xy[1], s=100, c="k", marker="o")

# plot line through the points in pos
plt.plot(
    [0, pos[0, 0].item(), pos[1, 0].item(), pos[2, 0].item()],
    [1, pos[0, 1].item(), pos[1, 1].item(), pos[2, 1].item()],
    color=(0, 0, 0, 1),
    linewidth=3,
)

plt.xlim(-1, 1)
plt.ylim(-3, 2)

ax.scatter(
    M_part[:, 0, 0],
    M_part[:, 0, 1],
    c="#EDAC32",
    alpha=1,
    edgecolor="none",
    s=15,
    zorder=-1,
)
ax.scatter(
    M_part[:, 1, 0],
    M_part[:, 1, 1],
    c="#D81B60",
    alpha=1,
    edgecolor="none",
    s=15,
    zorder=-1,
)
ax.scatter(
    M_part[:, 2, 0],
    M_part[:, 2, 1],
    c="#1E88E5",
    alpha=1,
    edgecolor="none",
    s=15,
    zorder=-1,
)

set_axis_color(ax, edgecolor)
ax.tick_params(direction=tick_direction, labelsize=20)

ax.set_title("CNF in distribution", fontsize=20)
sns.despine()
plt.savefig("./pendulum_cnf_in.png", bbox_inches="tight")
plt.savefig("./pendulum_cnf_in.pdf", bbox_inches="tight")
plt.show()
