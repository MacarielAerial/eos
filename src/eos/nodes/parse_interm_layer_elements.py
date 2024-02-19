import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from eos.data_interfaces.edge_dfs_data_interface import (
    EdgeAttrKey,
    EdgeDF,
    EdgeDFs,
    EdgeType,
)
from eos.data_interfaces.node_dfs_data_interface import (
    NodeAttrKey,
    NodeDF,
    NodeDFs,
    NodeType,
)

logger = logging.getLogger(__name__)


def parse_sub_industry_nodes(sub_industry_label: np.ndarray) -> NodeDF:
    # Sub industry label array by default is of the same shape
    # as theme array
    unique_label = np.unique(sub_industry_label)

    # During node dataframe aggregation, node ids will be reassigned
    nid = np.arange(len(unique_label))

    ntype = np.array([NodeType.sub_industry.value] * len(nid))

    df_sub_industry = pd.DataFrame(
        {
            NodeAttrKey.nid.value: nid,
            NodeAttrKey.ntype.value: ntype,
            NodeAttrKey.label.value: unique_label,
        }
    )

    logger.info(
        "Parsed a sub industry node dataframe shaped "
        f"{df_sub_industry.shape} from a sub industry label "
        f"array shaped {sub_industry_label.shape}"
    )

    return NodeDF(ntype=NodeType.sub_industry, df=df_sub_industry)


def parse_industry_nodes(industry_label: np.ndarray) -> NodeDF:
    # Industry label array is of the same shape as sub industry array
    unique_label = np.unique(industry_label)

    # Placeholders
    nid = np.arange(len(unique_label))

    ntype = np.array([NodeType.industry.value] * len(nid))

    df_industry = pd.DataFrame(
        {
            NodeAttrKey.nid.value: nid,
            NodeAttrKey.ntype.value: ntype,
            NodeAttrKey.label.value: unique_label,
        }
    )

    logger.info(
        f"Parsed an industry node dataframe shaped {df_industry.shape} "
        f"from an industry label array shaped {industry_label.shape}"
    )

    return NodeDF(ntype=NodeType.industry, df=df_industry)


def reassign_consecutive_nid(node_df_reassign: NodeDF, node_dfs: NodeDFs) -> NodeDF:
    # Obtain all existing node ids
    existing_nid: List[int] = []
    for node_df in node_dfs.members:
        df = node_df.df
        existing_nid.extend(df[NodeAttrKey.nid.value].tolist())

    # Increment to-be-reassigned node dataframe's nid column values
    max_nid = max(existing_nid)
    node_df_reassign.df[NodeAttrKey.nid.value] = (
        node_df_reassign.df[NodeAttrKey.nid.value].to_numpy() + max_nid + 1
    )

    logger.info(
        f"Incremented {node_df_reassign.ntype.value} node ids by " f"{max_nid + 1}"
    )

    return node_df_reassign


def return_index_of_uniques_given_instances(
    uniques: np.ndarray, instances: np.ndarray
) -> np.ndarray:
    if len(uniques) != len(np.unique(uniques)):
        raise ValueError("Input uniques array is not unique")

    if not set(instances) <= set(uniques):
        raise ValueError("At least one instance value does not exist in uniques array")

    # Get sorted index of the array with unique values
    sorted_indices = np.argsort(uniques)

    # Find index of unique values in the order of instances
    indices_in_sorted = np.searchsorted(uniques[sorted_indices], instances)

    # Map the index back to the order of unique values
    final_indices = sorted_indices[indices_in_sorted]

    return final_indices


def parse_theme_to_sub_industry_edges(
    df_theme: DataFrame, df_sub_industry: DataFrame, sub_industry_label: np.ndarray
) -> EdgeDF:
    # Get index of sub industry destination node labels
    index_dst_nid = return_index_of_uniques_given_instances(
        df_sub_industry[NodeAttrKey.label.value].to_numpy(), sub_industry_label
    )
    # Use this index to obtain sub industry node ids
    dst_nid = df_sub_industry[NodeAttrKey.nid.value].to_numpy()[index_dst_nid]

    # Sub industry node ids are assumed to have been reassigned
    # to avoid duplicate node ids
    eid_ndarray = np.column_stack((df_theme[NodeAttrKey.nid.value].to_numpy(), dst_nid))
    eid: List[Tuple[int, int]] = list(map(tuple, eid_ndarray))

    etype = np.array([EdgeType.theme_to_sub_industry.value] * len(eid))

    df_theme_to_sub_industry = pd.DataFrame(
        {EdgeAttrKey.eid.value: eid, EdgeAttrKey.etype.value: etype}
    )

    logger.info(
        "Parsed a theme to sub-industry edge dataframe shaped "
        f"{df_theme_to_sub_industry.shape} from a theme dataframe shaped "
        f"{df_theme.shape}, a sub-industry dataframe shaped "
        f"{df_sub_industry.shape} and a sub industry label array "
        f"shaped {sub_industry_label.shape}"
    )

    return EdgeDF(etype=EdgeType.theme_to_sub_industry, df=df_theme_to_sub_industry)


def parse_sub_industry_to_industry_edges(
    df_sub_industry: DataFrame, df_industry: DataFrame, industry_label: np.ndarray
) -> EdgeDF:
    # Get index of industry destination node labels
    index_dst_nid = return_index_of_uniques_given_instances(
        df_industry[NodeAttrKey.label.value].to_numpy(), industry_label
    )
    # Use this index to obtain industry node ids
    dst_nid = df_industry[NodeAttrKey.nid.value].to_numpy()[index_dst_nid]

    # Industry node ids are assumed to have been reassigned
    # to avoid duplicate node ids
    eid_ndarray = np.column_stack(
        (df_sub_industry[NodeAttrKey.nid.value].to_numpy(), dst_nid)
    )
    eid: List[Tuple[int, int]] = list(map(tuple, eid_ndarray))

    etype = np.array([EdgeType.sub_industry_to_industry.value] * len(eid))

    df_sub_industry_to_industry = pd.DataFrame(
        {EdgeAttrKey.eid.value: eid, EdgeAttrKey.etype.value: etype}
    )

    logger.info(
        "Parsed a sub-industry to industry edge dataframe shaped "
        f"{df_sub_industry_to_industry.shape} from a sub-industry dataframe "
        f"shaped {df_sub_industry.shape}, an industry dataframe shaped "
        f"{df_industry.shape} and an industry label array "
        f"shaped {industry_label.shape}"
    )

    return EdgeDF(
        etype=EdgeType.sub_industry_to_industry, df=df_sub_industry_to_industry
    )


# TODO: The following function is a placeholder for more complicated logic
# which should trace Industry nodes to their source Theme nodes before
# linking Industry nodes to different Sector nodes
def parse_industry_to_sector_edges(
    df_industry: DataFrame, df_sector: DataFrame
) -> EdgeDF:
    # Use heuristics to link all industry nodes to all sector nodes
    # The assumption is there's only one sector node
    nid_src = df_industry[NodeAttrKey.nid.value].to_list()
    nid_dst = df_sector[NodeAttrKey.nid.value].to_list()
    eid: List[Tuple[int, int]] = list(zip(nid_src, nid_dst * len(nid_src)))

    etype = np.array([EdgeType.industry_to_sector.value] * len(eid))

    df_industry_to_sector = pd.DataFrame(
        {EdgeAttrKey.eid.value: eid, EdgeAttrKey.etype.value: etype}
    )

    logger.info(
        "Parsed a industry to sector edge dataframe shaped "
        f"{df_industry_to_sector.shape} from an industry dataframe "
        f"shaped {df_industry.shape}, a sector dataframe shaped "
        f"{df_sector.shape}"
    )

    return EdgeDF(etype=EdgeType.industry_to_sector, df=df_industry_to_sector)


def parse_interm_layer_node_dfs(
    base_node_dfs: NodeDFs, sub_industry_label: np.ndarray, industry_label: np.ndarray
) -> NodeDFs:
    # Parse intermediate layer node dataframes first with conflicting node ids
    sub_industry_node_df = parse_sub_industry_nodes(
        sub_industry_label=sub_industry_label
    )
    industry_node_df = parse_industry_nodes(industry_label=industry_label)

    # Reassign intermediate layer node ids to resolve node id conflict
    sub_industry_node_df = reassign_consecutive_nid(
        node_df_reassign=sub_industry_node_df, node_dfs=base_node_dfs
    )

    base_node_dfs.members.append(sub_industry_node_df)  # Iteratively resolve conflict
    base_node_dfs.validate()

    industry_node_df = reassign_consecutive_nid(
        node_df_reassign=industry_node_df, node_dfs=base_node_dfs
    )

    base_node_dfs.members.append(industry_node_df)
    base_node_dfs.validate()

    logger.info(
        f"{len(base_node_dfs.members)} node dataframes exist after "
        "parsing intermediate layer nodes"
    )

    return base_node_dfs


def parse_interm_layer_edge_dfs(
    interm_node_dfs: NodeDFs,
    base_edge_dfs: EdgeDFs,
    sub_industry_label: np.ndarray,
    industry_label: np.ndarray,
) -> EdgeDFs:
    # Intermediate node dataframes are assumed to have already been parsed
    # Parse intermediate layer edge dataframes
    ntype_to_df = interm_node_dfs.to_dict()

    theme_to_sub_industry = parse_theme_to_sub_industry_edges(
        df_theme=ntype_to_df[NodeType.theme],
        df_sub_industry=ntype_to_df[NodeType.sub_industry],
        sub_industry_label=sub_industry_label,
    )

    base_edge_dfs.members.append(theme_to_sub_industry)
    base_edge_dfs.validate()

    sub_industry_to_industry = parse_sub_industry_to_industry_edges(
        df_sub_industry=ntype_to_df[NodeType.sub_industry],
        df_industry=ntype_to_df[NodeType.industry],
        industry_label=industry_label,
    )

    base_edge_dfs.members.append(sub_industry_to_industry)
    base_edge_dfs.validate()

    industry_to_sector = parse_industry_to_sector_edges(
        df_industry=ntype_to_df[NodeType.industry],
        df_sector=ntype_to_df[NodeType.sector],
    )

    base_edge_dfs.members.append(industry_to_sector)
    base_edge_dfs.validate()

    logger.info(
        f"{len(base_edge_dfs.members)} edge dataframes exist after "
        "parsing intermediate layer edges"
    )

    return base_edge_dfs


def _parse_interm_layer_elements(
    base_node_dfs: NodeDFs,
    base_edge_dfs: EdgeDFs,
    sub_industry_label: np.ndarray,
    industry_label: np.ndarray,
) -> Tuple[NodeDFs, EdgeDFs]:
    interm_node_dfs = parse_interm_layer_node_dfs(
        base_node_dfs=base_node_dfs,
        sub_industry_label=sub_industry_label,
        industry_label=industry_label,
    )

    interm_edge_dfs = parse_interm_layer_edge_dfs(
        interm_node_dfs=interm_node_dfs,
        base_edge_dfs=base_edge_dfs,
        sub_industry_label=sub_industry_label,
        industry_label=industry_label,
    )

    return interm_node_dfs, interm_edge_dfs
