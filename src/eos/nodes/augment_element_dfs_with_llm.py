import logging
from typing import Tuple

import pandas as pd
from pandas import DataFrame

from eos.data_interfaces.node_dfs_data_interface import NodeAttrKey, NodeDF

logger = logging.getLogger(__name__)


def augment_df_clusters(
    df_clusters: DataFrame, df_clusters_eval: DataFrame
) -> DataFrame:
    """df_clusters is either a sub industry node dataframe or an industry node "
    "dataframe which are both results of clustering"""
    # The key for this left join refers to integer cluster labels
    df = pd.merge(df_clusters, df_clusters_eval, on=NodeAttrKey.label.value, how="left")

    logger.info(
        f"Augmented cluster node type node dataframe shaped {df_clusters.shape} with LLM generated clusters evaluation dataframe shaped {df_clusters_eval.shape}"
    )

    return df


def _augment_element_dfs_with_llm(
    sub_industry_node_df: NodeDF,
    industry_node_df: NodeDF,
    df_sub_industry_eval: DataFrame,
    df_industry_eval: DataFrame,
) -> Tuple[NodeDF, NodeDF]:
    logger.info(
        "Augmenting sub industry node dataframe "
        "with text labels and evaluation nodes"
    )
    sub_industry_node_df.df = augment_df_clusters(
        df_clusters=sub_industry_node_df.df, df_clusters_eval=df_sub_industry_eval
    )

    logger.info(
        "Augmenting sub industry node dataframe "
        "with text labels and evaluation nodes"
    )
    industry_node_df.df = augment_df_clusters(
        df_clusters=industry_node_df.df, df_clusters_eval=df_industry_eval
    )

    return sub_industry_node_df, industry_node_df
