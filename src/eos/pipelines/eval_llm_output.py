import logging
from pathlib import Path

import orjson

from eos.data_interfaces.clusters_eval_data_interface import ClustersEvalDataInterface
from eos.nodes.eval_llm_output import _eval_llm_output

logger = logging.getLogger(__name__)


def eval_llm_output(
    path_sub_industry_clusters: Path,
    path_industry_clusters: Path,
    path_sub_industry_eval: Path,
    path_industry_eval: Path,
) -> None:
    # Data Access - Input
    with open(path_sub_industry_clusters, "rb") as f:
        sub_industry_clusters = orjson.loads(f.read())

        logger.info(
            f"Loaded a LLM generated raw sub industry clusters evaluation json from {path_sub_industry_clusters}"
        )

    with open(path_industry_clusters, "rb") as f:
        industry_clusters = orjson.loads(f.read())

        logger.info(
            f"Loaded a LLM generated raw industry clusters evaluation json from {path_industry_clusters}"
        )

    # Task Processing
    sub_industry_eval, industry_eval = _eval_llm_output(
        sub_industry_clusters=sub_industry_clusters, industry_clusters=industry_clusters
    )

    # Data Access - Output
    sub_industry_eval_data_interface = ClustersEvalDataInterface(
        filepath=path_sub_industry_eval
    )
    sub_industry_eval_data_interface.save(clusters_eval=sub_industry_eval)

    industry_eval_data_interface = ClustersEvalDataInterface(
        filepath=path_industry_eval
    )
    industry_eval_data_interface.save(clusters_eval=industry_eval)


if __name__ == "__main__":
    import argparse

    from eos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(description="Type LLM generated jsons")
    parser.add_argument(
        "-psic",
        "--path_sub_industry_clusters",
        type=Path,
        required=True,
        help="Path from which a LLM generated json evaluating "
        "sub industry level clustering is loaded",
    )
    parser.add_argument(
        "-pic",
        "--path_industry_clusters",
        type=Path,
        required=True,
        help="Path from which a LLM generated json evaluating "
        "industry level clustering is loaded",
    )
    parser.add_argument(
        "-psie",
        "--path_sub_industry_eval",
        type=Path,
        required=True,
        help="Path to which typed sub industry clusters evaluation is saved",
    )
    parser.add_argument(
        "-pie",
        "--path_industry_eval",
        type=Path,
        required=True,
        help="Path to which typed industry clusters evaluation is saved",
    )

    args = parser.parse_args()

    eval_llm_output(
        path_sub_industry_clusters=args.path_sub_industry_clusters,
        path_industry_clusters=args.path_industry_clusters,
        path_sub_industry_eval=args.path_sub_industry_eval,
        path_industry_eval=args.path_industry_eval,
    )
