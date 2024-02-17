import numpy as np
import pandas as pd
from pytest import fixture

from eos.data_interfaces.edge_dfs_data_interface import EdgeAttrKey, EdgeDF, EdgeType
from eos.data_interfaces.node_dfs_data_interface import (
    NodeAttrKey,
    NodeDF,
    NodeDFs,
    NodeType,
)
from eos.nodes.parse_interm_layer_elements import (
    parse_industry_nodes,
    parse_sub_industry_nodes,
    parse_sub_industry_to_industry_edges,
    parse_theme_to_sub_industry_edges,
    reassign_consecutive_nid,
)


@fixture
def mock_industry_label() -> np.ndarray:
    return np.array(["Tech", "Health", "Tech", "Auto"])


@fixture
def mock_sub_industry_label() -> np.ndarray:
    return np.array(["Software", "Pharmaceuticals", "Hardware", "Electric"])


def test_parse_sub_industry_nodes(mock_sub_industry_label: np.ndarray) -> None:
    node_df = parse_sub_industry_nodes(mock_sub_industry_label)

    assert isinstance(node_df, NodeDF)
    assert node_df.ntype == NodeType.sub_industry
    assert len(node_df.df) == len(np.unique(mock_sub_industry_label))
    assert all(
        node_df.df.columns
        == [NodeAttrKey.nid.value, NodeAttrKey.ntype.value, NodeAttrKey.label.value]
    )


def test_parse_industry_nodes(mock_industry_label: np.ndarray) -> None:
    node_df = parse_industry_nodes(mock_industry_label)
    assert isinstance(node_df, NodeDF)
    assert node_df.ntype == NodeType.industry
    assert len(node_df.df) == len(np.unique(mock_industry_label))
    assert all(
        node_df.df.columns
        == [NodeAttrKey.nid.value, NodeAttrKey.ntype.value, NodeAttrKey.label.value]
    )


def test_reassign_consecutive_nid() -> None:
    initial_node_df = NodeDF(
        ntype=NodeType.industry,
        df=pd.DataFrame(
            {NodeAttrKey.nid.value: [0, 1], NodeAttrKey.label.value: ["Tech", "Auto"]}
        ),
    )
    node_dfs = NodeDFs(members=[initial_node_df])

    new_node_df = NodeDF(
        ntype=NodeType.sub_industry,
        df=pd.DataFrame(
            {
                NodeAttrKey.nid.value: [0, 1],
                NodeAttrKey.label.value: ["Software", "Electric"],
            }
        ),
    )
    reassigned_node_df = reassign_consecutive_nid(new_node_df, node_dfs)

    assert (
        reassigned_node_df.df[NodeAttrKey.nid.value].min()
        > initial_node_df.df[NodeAttrKey.nid.value].max()
    )


def test_parse_theme_to_sub_industry_edges() -> None:
    # Mock data for theme and sub-industry nodes
    df_theme = pd.DataFrame(
        {NodeAttrKey.nid.value: [0, 1], NodeAttrKey.label.value: ["Theme1", "Theme2"]}
    )
    df_sub_industry = pd.DataFrame(
        {
            NodeAttrKey.nid.value: [10, 11],
            NodeAttrKey.label.value: ["SubIndustry1", "SubIndustry2"],
        }
    )
    sub_industry_label = np.array(["SubIndustry2", "SubIndustry1"])

    # Expected edges: Theme1->SubIndustry2, Theme2->SubIndustry1
    edge_df = parse_theme_to_sub_industry_edges(
        df_theme, df_sub_industry, sub_industry_label
    )
    assert isinstance(edge_df, EdgeDF)
    assert edge_df.etype == EdgeType.theme_to_sub_industry
    assert len(edge_df.df) == len(sub_industry_label)
    assert all(edge_df.df.columns == [EdgeAttrKey.eid.value, EdgeAttrKey.etype.value])
    assert (
        edge_df.df[EdgeAttrKey.etype.value].unique()[0]
        == EdgeType.theme_to_sub_industry.value
    )


def test_parse_sub_industry_to_industry_edges() -> None:
    # Mock data for sub-industry and industry nodes
    df_sub_industry = pd.DataFrame(
        {
            NodeAttrKey.nid.value: [0, 1],
            NodeAttrKey.label.value: ["SubIndustry1", "SubIndustry2"],
        }
    )
    df_industry = pd.DataFrame(
        {
            NodeAttrKey.nid.value: [10, 11],
            NodeAttrKey.label.value: ["Industry1", "Industry2"],
        }
    )
    industry_label = np.array(["Industry2", "Industry1"])

    # Expected edges: SubIndustry1->Industry2, SubIndustry2->Industry1
    edge_df = parse_sub_industry_to_industry_edges(
        df_sub_industry, df_industry, industry_label
    )
    assert isinstance(edge_df, EdgeDF)
    assert edge_df.etype == EdgeType.sub_industry_to_industry
    assert len(edge_df.df) == len(industry_label)
    assert all(edge_df.df.columns == [EdgeAttrKey.eid.value, EdgeAttrKey.etype.value])
    assert (
        edge_df.df[EdgeAttrKey.etype.value].unique()[0]
        == EdgeType.sub_industry_to_industry.value
    )
