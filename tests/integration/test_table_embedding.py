"""
Tests whether table_embedding pipeline can successfully convert strng-letters
into their numeric representation in vector space
"""

import random
from typing import Dict, Any

import pandas as pd
from pandas import DataFrame

from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from eos.refinery.encode_table.encode_table import TableEncoder
from eos.refinery.encode_table.autoencoder import AutoEncoder, TorchDataset, PreprocessEncoder

import logging
import yaml

from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from pipelinex import FlexiblePipeline, HatchDict

from eos.utils import get_feed_dict


conf_loader: ConfigLoader = ConfigLoader(conf_paths = ["eos/conf/base", "eos/conf/local"])

conf_logging: Dict[str, Any] = conf_loader.get("logging*", "logging*/**")
logging.config.dictConfig(conf_logging)

conf_catalog: Dict[str, Any] = conf_loader.get("catalog*", "catalog*/**")
data_catalog: DataCatalog = DataCatalog.from_config(conf_catalog)

conf_params: Dict[str, Any] = conf_loader.get("parameters*", "parameters*/**")
data_catalog.add_feed_dict(feed_dict=get_feed_dict(params=conf_params))

conf_pipeline: Dict[str, Any] = conf_loader.get("pipelines*", "pipelines*/**")

runner: SequentialRunner = SequentialRunner()

def test_train_autoencoder() -> None:
    pipeline: FlexiblePipeline = HatchDict(conf_pipeline).get("autoencoder_pipeline")

    runner.run(pipeline = pipeline, catalog = data_catalog)

    assert False

def test_embed_table() -> None:
    df_raw: DataFrame = DataFrame({"cont_1": [1.0, 2.0, 3.0],
                                 "cat_1": ["Los Angeles", "New York", "Austin"],
                                 "cat_2": ["Los Angeles Dodgers", "New York Yankees", "Round Rock Express"]})
    df_raw.attrs.update({"cat_feats": ["cat_1", "cat_2"], "cont_feats": ["cont_1"]})
    te_obj: TableEncoder = TableEncoder(df_input=df_raw)
    te_obj.split_cat_cont()

    assert False
