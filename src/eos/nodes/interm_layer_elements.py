import logging
from typing import List, Tuple
from h11 import Data

import numpy as np
import pandas as pd
from pandas import DataFrame
from eos.data_interfaces.edge_dfs_data_interface import EdgeAttrKey, EdgeType

from eos.data_interfaces.node_dfs_data_interface import NodeAttrKey, NodeDF, NodeType

logger = logging.getLogger(__name__)


def parse_sub_industry_nodes(sub_industry_label: np.ndarray) -> NodeDF:
    # Sub industry label array by default is of the same shape
    # as theme array
    unique_label = np.unique(sub_industry_label)

    # During node dataframe aggregation, node ids will be reassigned
    nid = np.arange(len(unique_label))

    ntype = np.array([NodeType.sub_industry.value] * len(nid))

    df_sub_industry = pd.DataFrame(
        {NodeAttrKey.nid.value: nid,
         NodeAttrKey.ntype.value: ntype,
         NodeAttrKey.label.value: unique_label}
    )

    logger.info("Parsed a sub industry node dataframe shaped "
                f"{df_sub_industry.shape} from a sub industry label "
                f"array shaped {sub_industry_label.shape}")

    return df_sub_industry

def parse_industry_nodes(industry_label: np.ndarray) -> NodeDF:
    # Industry label array is of the same shape as sub industry array
    unique_label = np.unique(industry_label)

    # Placeholders
    nid = np.arange(len(unique_label))
    
    ntype = np.array([NodeType.industry.value] * len(nid))

    df_industry = pd.DataFrame(
        {NodeAttrKey.nid.value: nid,
         NodeAttrKey.ntype.value: ntype,
         NodeAttrKey.label.value: unique_label}
    )

    logger.info(f"Parsed an industry node dataframe shaped {df_industry.shape} "
                f"from an industry label array shaped {industry_label.shape}")
    
    return df_industry

def return_index_of_uniques_given_instances(uniques: np.ndarray, instances: np.ndarray) -> np.ndarray:
    # Get sorted index of the array with unique values
    sorted_indices = np.argsort(uniques)

    # Find index of unique values in the order of instances
    indices_in_sorted = np.searchsorted(uniques[sorted_indices], instances)

    # Map the index back to the order of unique values
    final_indices = sorted_indices[indices_in_sorted]

    return final_indices

def parse_theme_to_sub_industry_edges(df_theme: DataFrame, df_sub_industry: DataFrame, sub_industry_label: np.ndarray) -> DataFrame:
    # Get index of sub industry destination node labels
    index_dst_nid = return_index_of_uniques_given_instances(df_sub_industry[NodeAttrKey.label.value].values, sub_industry_label)
    # Use this index to obtain sub industry node ids
    dst_nid = df_sub_industry[NodeAttrKey.nid.value].values[index_dst_nid]

    # Sub industry node ids are assumed to have been reassigned
    # to avoid duplicate node ids
    eid_ndarray = np.column_stack((df_theme[NodeAttrKey.nid.value].values, dst_nid))
    eid: List[Tuple[int, int]] = list(map(tuple, eid_ndarray))

    etype = np.array([EdgeType.theme_to_sub_industry.value] * len(eid))

    df_theme_to_sub_industry = pd.DataFrame(
        {EdgeAttrKey.eid.value: eid,
         EdgeAttrKey.etype.value: etype}
    )

    logger.info("Parsed a theme to sub-industry edge dataframe shaped "
                f"{df_theme_to_sub_industry.shape} from a theme dataframe shaped "
                f"{df_theme.shape}, a sub-industry dataframe shaped "
                f"{df_sub_industry.shape} and a sub industry label array "
                f"shaped {sub_industry_label.shape}")

    return df_theme_to_sub_industry

def parse_sub_industry_to_industry_edges(df_sub_industry: DataFrame, df_industry: DataFrame, industry_label: np.ndarray) -> DataFrame:
    # Get index of industry destination node labels
    index_dst_nid = return_index_of_uniques_given_instances(df_industry[NodeAttrKey.label.value].values, industry_label)
    # Use this index to obtain industry node ids
    dst_nid = df_industry[NodeAttrKey.nid.value].values[index_dst_nid]

    # Industry node ids are assumed to have been reassigned
    # to avoid duplicate node ids
    eid_ndarray = np.column_stack((df_sub_industry[NodeAttrKey.nid.value].values, dst_nid))
    eid: List[Tuple[int, int]] = list(map(tuple, eid_ndarray))

    etype = np.array([EdgeType.sub_industry_to_industry.value] * len(eid))

    df_sub_industry_to_industry = pd.DataFrame(
        {EdgeAttrKey.eid.value: eid,
         EdgeAttrKey.etype.value: etype}
    )

    logger.info("Parsed a sub-industry to industry edge dataframe shaped "
                f"{df_sub_industry_to_industry.shape} from a sub-industry dataframe "
                f"shaped {df_sub_industry.shape}, an industry dataframe shaped "
                f"{df_industry.shape} and an industry label array "
                f"shaped {industry_label.shape}")

    return df_sub_industry_to_industry
