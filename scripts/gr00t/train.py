import tyro
from loguru import logger
from dataclasses import dataclass


@dataclass
class Config:
    """
    If no indexes_to_delete is provided, will attempt to repair the dataset.
    If indexes_to_delete is provided, will delete the specified indexes from the dataset and repair the rest.
    """

    dataset_id: str
    """the id of the dataset to train on"""


if __name__ == "__main__":
    config = tyro.cli(Config)

    logger.info(f"Starting training for dataset={config.dataset_id}")
