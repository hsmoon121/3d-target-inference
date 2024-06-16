import os, datetime
import wandb

class Logger(object):
    def __init__(self, name, last_step=0, board=True, board_path="./data/board"):
        self.name = name
        self.train_step = last_step
        run = wandb.init(
            project="amortized-inference",
            name=name,
        )
    def step(self):
        self.train_step += 1

    def write_scalar(self, verbose=False, **kwargs):
        wandb.log(kwargs, step=self.train_step)