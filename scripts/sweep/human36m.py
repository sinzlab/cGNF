from propose.datasets.human36m.Human36mDataset import Human36mDataset

from torch_geometric.loader import DataLoader

from propose.models.flows import CondGraphFlow
from propose.models.nn.embedding import embeddings
from propose.training import supervised_trainer
from propose.utils.reproducibility import set_random_seed

import torch

import wandb


def build_config(config, sweep_config):
    # model config
    config["model"]["num_layers"] = sweep_config["num_layers"]
    config["model"]["context_features"] = sweep_config["embedding_out_features"]
    config["model"]["hidden_features"] = sweep_config["hidden_features"]

    config["embedding"]["config"]["hidden_dim"] = sweep_config[
        "embedding_hidden_features"
    ]
    config["embedding"]["config"]["output_dim"] = sweep_config["embedding_out_features"]

    return config


def human36m(
    use_wandb: bool = False,
    config: dict = None,
):
    """
    Train a CondGraphFlow on the Human36m dataset.
    :param use_wandb: Whether to use wandb for logging.
    :param config: A dictionary of configuration parameters.
    :param train_config_file: A dictionary of training configuration parameters.
    """
    wandb.init()

    sweep_config = wandb.config
    config = build_config(config, sweep_config)
    wandb.config.update(config)

    set_random_seed(config["seed"])

    dataset = Human36mDataset(**config["dataset"])

    dataloader = DataLoader(
        dataset, batch_size=config["train"]["batch_size"], shuffle=True
    )

    embedding_net = None
    if config["embedding"]:
        embedding_net = embeddings[config["embedding"]["name"]](
            **config["embedding"]["config"]
        )

    flow = CondGraphFlow(**config["model"], embedding_net=embedding_net)

    num_params = sum(p.numel() for p in flow.parameters())
    print(f"Number of parameters: {num_params}")

    # set number of parameters in wandb config
    if use_wandb:
        wandb.config.num_params = num_params

    if "use_pretrained" in config:
        artifact = wandb.run.use_artifact(
            f'ppierzc/propose_human36m/{config["use_pretrained"]}', type="model"
        )
        artifact_dir = artifact.download()
        flow.load_state_dict(torch.load(artifact_dir + "/model.pt"))

    if config["cuda_accelerated"]:
        flow.to("cuda:0")

    optimizer = torch.optim.Adam(flow.parameters(), **config["train"]["optimizer"])

    lr_scheduler = None
    if config["train"]["lr_scheduler"]:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config["train"]["lr_scheduler"], verbose=True
        )

    supervised_trainer(
        dataloader,
        flow,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=config["train"]["epochs"],
        device=flow.device,
        use_wandb=use_wandb,
    )
