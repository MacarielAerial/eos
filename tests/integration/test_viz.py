"""
Tests custom pipeline visualisation function
"""

import logging
from pathlib import Path
from typing import Any, Dict

import requests
from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from pipelinex import FlexiblePipeline, HatchDict

from eos.office.viz import call_viz
from eos.utils import get_feed_dict

log = logging.getLogger(__name__)


def test_viz() -> None:
    dir_static_site: str = "./public"
    # Configure pipeline and catalog objects
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
    nx_pipeline: FlexiblePipeline = HatchDict(conf_pipeline).get("networkx_pipeline")
    dgl_pipeline: FlexiblePipeline = HatchDict(conf_pipeline).get("dgl_pipeline")

    pipelines: Dict[str, FlexiblePipeline] = {
        "autoencoder_pipeline": ae_pipeline,
        "networkx_pipeline": nx_pipeline,
        "dgl_pipeline": dgl_pipeline,
        "master_pipeline": ae_pipeline + nx_pipeline + dgl_pipeline,
    }
    # Parse Python object information into JSON form and export to local
    call_viz(dir_static_site=dir_static_site, catalog=data_catalog, pipelines=pipelines)
    # Serve the static website from local
    # run_static_server(directory = dir_static_site, port = 4141)
    assert Path(dir_static_site).joinpath("pipeline.json")


def _check_viz_up(port):
    url = "http://127.0.0.1:{}/".format(port)
    try:
        response = requests.get(url)
    except requests.ConnectionError:
        return False

    return response.status_code == 200
