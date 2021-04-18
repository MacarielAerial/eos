"""
Tests the end-to-end workflow for the package
"""

import logging
from typing import Any, Dict

from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from pipelinex import FlexiblePipeline, HatchDict

from eos.utils import get_feed_dict

log = logging.getLogger(__name__)


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
    # ae_pipeline: FlexiblePipeline = HatchDict(conf_pipeline).get("autoencoder_pipeline")
    # nx_pipeline: FlexiblePipeline = HatchDict(conf_pipeline).get("networkx_pipeline")
    dgl_pipeline: FlexiblePipeline = HatchDict(conf_pipeline).get("dgl_pipeline")

    runner: SequentialRunner = SequentialRunner()
    runner.run(pipeline=dgl_pipeline, catalog=data_catalog)

    assert False
