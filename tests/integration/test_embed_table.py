"""
Tests whether categorical columns of a table can be embedded
with an autoencoder
"""

import logging
from typing import Any, Dict

from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from pipelinex import FlexiblePipeline, HatchDict

from eos.utils import get_feed_dict


def test_e2e() -> None:
    conf_loader: ConfigLoader = ConfigLoader(
        conf_paths=["eos/conf/base", "eos/conf/local"]
    )

    conf_logging: Dict[str, Any] = conf_loader.get("logging*", "logging*/**")
    logging.config.dictConfig(conf_logging)

    conf_catalog: Dict[str, Any] = conf_loader.get("catalog*", "catalog*/**")
    data_catalog: DataCatalog = DataCatalog.from_config(conf_catalog)

    conf_params: Dict[str, Any] = conf_loader.get("parameters*", "parameters*/**")
    data_catalog.add_feed_dict(feed_dict=get_feed_dict(params=conf_params))

    conf_pipeline: Dict[str, Any] = conf_loader.get("pipelines*", "pipelines*/**")
    ae_pipeline: FlexiblePipeline = HatchDict(conf_pipeline).get("autoencoder_pipeline")

    runner: SequentialRunner = SequentialRunner()
    runner.run(pipeline=ae_pipeline, catalog=data_catalog)
