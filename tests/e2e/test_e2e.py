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
    nx_pipeline: FlexiblePipeline = HatchDict(conf_pipeline).get("networkx_pipeline")
    dgl_pipeline: FlexiblePipeline = HatchDict(conf_pipeline).get("dgl_pipeline")

    runner: SequentialRunner = SequentialRunner()
    runner.run(pipeline=ae_pipeline + nx_pipeline + dgl_pipeline, catalog=data_catalog)

    assert False


"""
def test_e2e_dep() -> None:
    # Test-specific parameter definitions
    path_catalog_yml: str = "tests/data/catalog.yml"
    catalog: dict = yaml.safe_load(open(path_catalog_yml, "r"))
    input_key: str = "test_e2e"

    # CSV Data access operations
    node_dataset: CSVDataSetE = HatchDict(catalog[input_key]).get("node_dataset")
    node_data: DataFrame = node_dataset.load()

    edge_dataset: CSVDataSetE = HatchDict(catalog[input_key]).get("edge_dataset")
    edge_data: DataFrame = edge_dataset.load()

    # Node population
    gc_obj: GraphCreator = GraphCreator(df_input=node_data)
    gc_obj.create_graph()

    # Node: NetworkX Data access operations
    nx_node_dataset: NetworkXDataSetE = HatchDict(catalog[input_key]).get(
        "nx_node_dataset"
    )
    nx_node_dataset.save(gc_obj.g)

    nx_node_g_reloaded: Graph = nx_node_dataset.load()

    # Edge linking
    nl_obj: NodeLinker = NodeLinker(g_input=nx_node_g_reloaded, df_input=edge_data)
    nl_obj.link_node()

    # Node with edge: NetworkX Data access operations
    nx_graph_dataset: NetworkXDataSetE = HatchDict(catalog[input_key]).get(
        "nx_graph_dataset"
    )
    nx_graph_dataset.save(nl_obj.g)

    nx_graph_g_reloaded: Graph = nx_graph_dataset.load()

    # Graph feature concatenation
    fe_obj: FeatureConcatenator = FeatureConcatenator(g_input=nx_graph_g_reloaded)
    fe_obj.concat_n_attrs()
    fe_obj.concat_e_attrs()

    # Concatenated graph: NetworkX Data access operations
    nx_concat_dataset: NetworkXDataSetE = HatchDict(catalog[input_key]).get(
        "nx_concat_dataset"
    )
    nx_concat_dataset.save(fe_obj.g)

    nx_concat_g_reloaded: Graph = nx_concat_dataset.load()

    # DGL Graph conversion
    dgl_g: DGLHeteroGraph = from_networkx(
        nx_concat_g_reloaded, node_attrs=["nfeat"], edge_attrs=["efeat"]
    )

    # DGL Data access operations
    dgl_dataset: DGLDataSet = HatchDict(catalog[input_key]).get("dgl_dataset")
    dgl_dataset.save(dgl_g)

    glist, label_dict = dgl_dataset.load()
    dgl_g_reloaded: DGLHeteroGraph = glist[0]

    assert dgl_g_reloaded.ndata
    assert dgl_g_reloaded.edata
"""
