"""
Tests whether categorical columns of a table can be embedded
with an autoencoder
"""

import logging
from typing import Any, Dict

from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from pandas import DataFrame
from pipelinex import FlexiblePipeline, HatchDict

from eos.refinery.embed_table import ordinally_encode_table
from eos.utils import get_feed_dict


def test_table_embedding() -> None:
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


def test_ordinally_encode_table() -> None:
    df: DataFrame = DataFrame(
        {"col_1": ["a", "b", "c"], "col_2": ["d", "e", "f"], "col_3": [1, 2, 3]}
    )
    df.attrs = {"cat_feats": ["col_1", "col_2"]}
    df_encoded, categories = ordinally_encode_table(df=df)

    assert df_encoded.loc[0, "col_1"] == 0.0
    assert len(categories) == 2
