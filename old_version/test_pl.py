import os
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch import optim, nn, utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import torch
from utils.python_utils import available_cpu_count, init_run


def get_model_config(config_file_path: str):
    # TODO
    # Depends on your script/model
    # Should be loaded from a yaml file
    return {'cumulative_bs': 256, 'lr': 1e-3, 'max_epochs': 5}


class DummyModel(pl.LightningModule):

    def __init__(self, lr: float = 1e-3):

        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

        self.lr = lr

    def training_step(self, batch, _):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':

    SEED = 42
    workers = available_cpu_count()
    offline = True  # log to wandb if False

    pl.seed_everything(SEED, workers=True)

    args = init_run()

    # base_path
    base_path = '/home/dserez/'

    # config variables (take from ENV)
    num_nodes = int(os.getenv('NODES'))
    gpus = torch.cuda.device_count()
    rank = int(os.getenv('NODE_RANK'))

    # paths
    params_conf = args.conf
    project_path = f'{base_path}my_project/'

    # resume training
    ckpt, wandb_id = args.ckpt, args.wandb_id

    run_name = f'{params_conf}'
    checkpoint_dir = f'{project_path}runs/{run_name}/'

    # check if resume or train from scratch
    if ckpt is not None and wandb_id is not None:
        resume = True
        checkpoint_path = f'{checkpoint_dir}{ckpt}'
    else:
        resume = False
        checkpoint_path = None

    # get model config
    m_conf = get_model_config(f'{project_path}confs/{params_conf}.yaml')

    # get training params
    local_bs = m_conf['cumulative_bs'] // (num_nodes * gpus)
    lr = m_conf['base_lr'] * m_conf['cumulative_bs']

    if resume:
        model = DummyModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    else:
        # init model
        model = DummyModel(lr=lr)

    # init callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename='{epoch:02d}', save_last=True,
                                          save_on_train_epoch_end=True)

    callbacks = [LearningRateMonitor(), checkpoint_callback]

    # logger (to be defined only on rank zero to avoid duplicates (wandb bug))
    if rank == 0:
        if resume:
            logger = WandbLogger(project='ssl', name=run_name, offline=offline, id=wandb_id, resume='must')
        else:
            logger = WandbLogger(project='ssl', name=run_name, offline=offline)
    else:
        logger = WandbLogger(project='ssl', name=run_name, offline=True)

    # init trainer
    strategy = DDPStrategy(find_unused_parameters=False, static_graph=True)

    trainer = pl.Trainer(strategy=strategy, accelerator='gpu', num_nodes=num_nodes, devices=gpus,
                         callbacks=callbacks, deterministic=True, logger=logger,
                         check_val_every_n_epoch=5, max_epochs=m_conf['max_epochs'])

    # setup data
    dataset = MNIST(root=project_path, download=True, transform=ToTensor())
    train_loader = utils.data.DataLoader(dataset)

    # print for sanity check
    print(f"gpus: {gpus}")
    print(f"workers: {workers}")
    print(f"local_bs: {local_bs}")
    print(f"cumulative_bs: {m_conf['cumulative_bs']}")

    trainer.fit(model, train_loader, ckpt_path=checkpoint_path)
