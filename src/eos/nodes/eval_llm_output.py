from typing import Any, Dict, Tuple

from eos.data_interfaces.clusters_eval_data_interface import (
    ClusteringLevel,
    ClustersEval,
)


def _eval_llm_output(
    sub_industry_clusters: Dict[str, Any], industry_clusters: Dict[str, Any]
) -> Tuple[ClustersEval, ClustersEval]:
    # Parse raw json data
    sub_industry_eval = ClustersEval.from_raw_dict(
        level=ClusteringLevel.sub_industry, raw_dict=sub_industry_clusters
    )
    industry_eval = ClustersEval.from_raw_dict(
        level=ClusteringLevel.industry, raw_dict=industry_clusters
    )

    return sub_industry_eval, industry_eval
