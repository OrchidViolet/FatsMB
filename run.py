
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from src.model import RecModel
from src.datamodule import RecDataModule


def cli_main():
    cli = LightningCLI(RecModel, RecDataModule)

if __name__ == '__main__':
    cli_main()