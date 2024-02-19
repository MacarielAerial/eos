import logging
from pprint import pformat
from typing import Dict, List, Tuple

import numpy as np
from pandas import DataFrame

from eos.data_interfaces.edge_dfs_data_interface import EdgeAttrKey
from eos.data_interfaces.node_dfs_data_interface import NodeAttrKey

logger = logging.getLogger(__name__)


def get_out_edge_view(eid: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Converts an array of edge tuples to a unique source node array and a list of target node arrays."""
    # Sort edge ids based on source node ids
    eid_sorted = eid[eid[:, 0].argsort()]

    # Obtain unique source node ids and their corresponding first occurrence in sorted edge ids
    unique_nid_source, index_first_instance = np.unique(
        eid_sorted[:, 0], return_index=True
    )

    # Split target node ids based on the prepared indices
    split_nid_target = np.split(eid_sorted[:, 1], index_first_instance[1:])

    return unique_nid_source, split_nid_target


def map_nid_to_attr_val(
    nid_src_or_dst: List[np.ndarray], nid: np.ndarray, attr_val: np.ndarray
) -> List[np.ndarray]:
    """Input is assumed to be related to one node type only"""
    # Input edge member node ids could be two-dimensional (e.g. groups of nodes)
    flat_nid_src_or_dst = np.concatenate(nid_src_or_dst).ravel()

    # Sort all node ids
    sorting_index = np.argsort(nid)
    sorted_nid = nid[sorting_index]

    # Find edge member node ids' index in sorted all node ids
    cross_index_on_sorted = np.searchsorted(sorted_nid, flat_nid_src_or_dst)

    # Map cross index back to original unsorted node ids
    cross_index = sorting_index[cross_index_on_sorted]

    # Obtain attribute values ordered by edge member node ids
    mapped_attr_val_flat = attr_val[cross_index]

    # Convert back to input array shape
    mapped_attr_val: List[np.ndarray] = []
    i = 0
    for member_array in nid_src_or_dst:
        length = len(member_array)
        mapped_attr_val.append(mapped_attr_val_flat[i : i + length])
        i += length

    return mapped_attr_val


def collect_input_for_sub_industry(
    df_theme_to_sub_industry: DataFrame, df_theme: DataFrame, df_sub_industry: DataFrame
) -> Dict[int, List[str]]:
    # Identify theme-to-sub-industry edge ids
    eid: np.ndarray = np.array(df_theme_to_sub_industry[EdgeAttrKey.eid.value].tolist())

    # Obtain reversed edge ids
    eid_reverse: np.ndarray = eid[:, [1, 0]]

    # Map sub industry node ids to their target theme node ids
    unique_nid_sub_industry, split_nid_theme = get_out_edge_view(eid=eid_reverse)
    # Retrieve associated sub industry labels
    label_nested = map_nid_to_attr_val(
        nid_src_or_dst=[unique_nid_sub_industry],
        nid=df_sub_industry[NodeAttrKey.nid.value].to_numpy(),
        attr_val=df_sub_industry[NodeAttrKey.label.value].to_numpy(),
    )
    label = label_nested[0]  # Only one dimension
    split_theme = map_nid_to_attr_val(
        nid_src_or_dst=split_nid_theme,
        nid=df_theme[NodeAttrKey.nid.value].to_numpy(),
        attr_val=df_theme[NodeAttrKey.theme.value].to_numpy(),
    )

    sub_industry_label_to_split_theme: Dict[int, List[str]] = {
        label[i].item(): split_theme[i].tolist() for i in range(len(label))
    }

    logger.info(
        f"Parsed {len(sub_industry_label_to_split_theme)} input subgraphs to LLM"
    )

    return sub_industry_label_to_split_theme


def collect_input_for_industry(
    df_sub_industry_to_industry: DataFrame,
    df_sub_industry: DataFrame,
    df_industry: DataFrame,
    sub_industry_label_to_split_theme: Dict[int, List[str]],
) -> Dict[int, List[List[str]]]:
    # Identify sub-industry-to-industry edge ids
    eid: np.ndarray = np.array(
        df_sub_industry_to_industry[EdgeAttrKey.eid.value].tolist()
    )

    # Obtain reversed edge ids
    eid_reverse: np.ndarray = eid[:, [1, 0]]

    # Map industry node ids to their target sub industry node ids
    unique_nid_industry, split_nid_sub_industry = get_out_edge_view(eid=eid_reverse)

    # Retrieve associated sub industry labels
    industry_label_nested = map_nid_to_attr_val(
        nid_src_or_dst=[unique_nid_industry],
        nid=df_industry[NodeAttrKey.nid.value].to_numpy(),
        attr_val=df_industry[NodeAttrKey.label.value].to_numpy(),
    )
    industry_label = industry_label_nested[0]  # only one dimension
    split_sub_industry_label = map_nid_to_attr_val(
        nid_src_or_dst=split_nid_sub_industry,
        nid=df_sub_industry[NodeAttrKey.nid.value].to_numpy(),
        attr_val=df_sub_industry[NodeAttrKey.label.value].to_numpy(),
    )

    # TODO: Vectorise the following logic
    # Retrieve member themes associated with each sub industry nodes
    theme: List[List[List[str]]] = []
    for i, group_label in enumerate(split_sub_industry_label):
        theme.append([])
        for label in group_label:
            theme[i].append(sub_industry_label_to_split_theme[label])

    label_to_twice_split_theme: Dict[int, List[List[str]]] = {
        industry_label[i].item(): theme[i] for i in range(len(industry_label))
    }

    logger.info(f"Parsed {len(label_to_twice_split_theme)} input subgraphs to LLM")

    return label_to_twice_split_theme


def build_system_prompt() -> str:
    prompt = (
        "You are an industry classification expert who assists with clustering "
        "output evaluation and cluster description writing. You understand there are four "
        "levels of industry classifications from top to bottom: sector, industry, "
        "sub-industry and theme. Sector is the broadest category representing major "
        "economic segments. Industry is a more specific category that groups companies "
        "based on similar operational characteristics within a sector. Sub-Industry "
        "is an even more detailed category that identifies particular niches or market "
        "segments within an industry. Theme is the most granular level, focusing on "
        "current trends, technologies, or business practices shaping industries. "
        "You understand industry and sub-industry are targets for clustering. "
        "They are not described with natural language like themes and sectors are. "
        "They only have integer cluster labels. "
        "You understand all your responses should take json format. Your response should "
        "take the following schema: {INT_CLUSTER_LABEL: {'text_label': "
        "ONE_PHRASE_TEXT_LABEL, 'note': QUALITY_ASSESSMENT_NOTE}}. "
        "Your note should be concise and to the point. "
        "Your note should not be more than three sentences long. "
        "Your note should be very short for good quality clusters and longer if otherwise."
    )

    return prompt


def build_sub_industry_message_prompt(
    sub_industry_label_to_split_theme: Dict[int, List[str]],
) -> str:
    prompt = (
        "Here's the clustering result for themes on sub industry level. "
        "Can you provide text labels of sub industry clusters and provide evaluation "
        f"note for each cluster?\n{pformat(sub_industry_label_to_split_theme)}"
    )

    return prompt


def build_industry_message_prompt(
    label_to_twice_split_theme: Dict[int, List[List[str]]],
) -> str:
    prompt = (
        "Here's the clustering result for sub industries on industry level. "
        "Can you provide text labels of industry clusters and provide evaluation "
        f"note for each cluster?\n{pformat(label_to_twice_split_theme)}"
    )

    return prompt
