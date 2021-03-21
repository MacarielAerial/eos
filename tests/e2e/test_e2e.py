"""
Tests the end-to-end workflow for the package
"""

import yaml

from pipelinex import HatchDict
from pandas import DataFrame
from networkx import Graph
from dgl import from_networkx
from dgl.heterograph import DGLHeteroGraph

from eos.refinery.create_graph import GraphCreator
from eos.refinery.link_node import NodeLinker

from eos.factory.concat_feature import FeatureConcatenator

from eos.warehouse.csv_dataset import CSVDataSetE
from eos.warehouse.networkx_dataset import NetworkXDataSetE
from eos.warehouse.dgl_dataset import DGLDataSet

def test_e2e() -> None:
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
    gc_obj: GraphCreator = GraphCreator(df_input = node_data)
    gc_obj.create_graph()

    # Node: NetworkX Data access operations
    nx_node_dataset: NetworkXDataSetE = HatchDict(catalog[input_key]).get("nx_node_dataset")
    nx_node_dataset.save(gc_obj.g)

    nx_node_g_reloaded: Graph = nx_node_dataset.load()

    # Edge linking
    nl_obj: NodeLinker = NodeLinker(g_input = nx_node_g_reloaded,
                                    df_input = edge_data)
    nl_obj.link_node()

    # Node with edge: NetworkX Data access operations
    nx_graph_dataset: NetworkXDataSetE = HatchDict(catalog[input_key]).get("nx_graph_dataset")
    nx_graph_dataset.save(nl_obj.g)

    nx_graph_g_reloaded: Graph = nx_graph_dataset.load()

    # Graph feature concatenation
    fe_obj: FeatureConcatenator = FeatureConcatenator(g_input = nx_graph_g_reloaded)
    fe_obj.concat_n_attrs()
    fe_obj.concat_e_attrs()

    # Concatenated graph: NetworkX Data access operations
    nx_concat_dataset: NetworkXDataSetE = HatchDict(catalog[input_key]).get("nx_concat_dataset")
    nx_concat_dataset.save(fe_obj.g)

    nx_concat_g_reloaded: Graph = nx_concat_dataset.load()

    # DGL Graph conversion
    dgl_g: DGLHeteroGraph = from_networkx(nx_concat_g_reloaded, node_attrs = ["nfeat"], edge_attrs = ["efeat"])

    # DGL Data access operations
    dgl_dataset: DGLDataSet = HatchDict(catalog[input_key]).get("dgl_dataset")
    dgl_dataset.save(dgl_g)

    glist, label_dict = dgl_dataset.load()
    dgl_g_reloaded: DGLHeteroGraph = glist[0]

    assert dgl_g_reloaded.ndata
    assert dgl_g_reloaded.edata
