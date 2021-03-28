"""
Tests whether table_embedding pipeline can successfully convert strng-letters
into their numeric representation in vector space
"""

import logging
import random
from typing import Any, Dict

import pandas as pd
import yaml
from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from pandas import DataFrame
from pipelinex import FlexiblePipeline, HatchDict
from torch import nn, optim
from torch.utils.data import DataLoader

from eos.utils import get_feed_dict


def test_train_autoencoder() -> None:
    key_catalog: str = "test_train_autoencoder"

    conf_loader: ConfigLoader = ConfigLoader(
        conf_paths=[
            "tests/data/integration/conf/base",
            "tests/data/integration/conf/local",
        ]
    )

    conf_logging: Dict[str, Any] = conf_loader.get("logging*", "logging*/**")
    logging.config.dictConfig(conf_logging)

    conf_catalog: Dict[str, Any] = conf_loader.get("catalog*", "catalog*/**")
    data_catalog: DataCatalog = DataCatalog.from_config(conf_catalog)

    conf_params: Dict[str, Any] = conf_loader.get("parameters*", "parameters*/**")
    data_catalog.add_feed_dict(feed_dict=get_feed_dict(params=conf_params))

    conf_pipeline: Dict[str, Any] = conf_loader.get("pipelines*", "pipelines*/**")
    pipeline: FlexiblePipeline = HatchDict(conf_pipeline).get("autoencoder_pipeline")

    runner: SequentialRunner = SequentialRunner()
    runner.run(pipeline=pipeline, catalog=data_catalog)

    assert False
