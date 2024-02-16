import logging
from pathlib import Path

import numpy as np

from eos.nodes.k_means_cluster import _cluster_for_sub_and_industries

logger = logging.getLogger(__name__)


def cluster_for_sub_and_industries(
    path_theme_encoding: Path,
    path_description_encoding: Path,
    path_sub_industry_label: Path,
    path_industry_label: Path,
) -> None:
    # Data Access - Input
    theme_encoding: np.ndarray = np.load(path_theme_encoding)
    logger.info(
        f"Loaded encoded themes shaped {theme_encoding.shape} "
        f"from {path_theme_encoding}"
    )

    description_encoding: np.ndarray = np.load(path_description_encoding)
    logger.info(
        f"Loaded encoded descriptions shaped {description_encoding.shape} "
        f"from {path_description_encoding}"
    )

    # Task Processing
    sub_industry_label, industry_label = _cluster_for_sub_and_industries(
        theme_encoding=theme_encoding, description_encoding=description_encoding
    )

    # Data Access - Output
    np.save(path_sub_industry_label, sub_industry_label)
    logger.info(
        f"Saved sub industry label shaped {sub_industry_label.shape} to "
        f"{path_sub_industry_label}"
    )

    np.save(path_industry_label, industry_label)
    logger.info(
        f"Saved industry label shaped {industry_label.shape} to "
        f"{path_industry_label}"
    )


if __name__ == "__main__":
    import argparse

    from eos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Clusters single sector themes into sub industries and industries"
    )
    parser.add_argument(
        "-pte",
        "--path_theme_encoding",
        type=Path,
        required=True,
        help="Path from which encoded themes are loaded",
    )
    parser.add_argument(
        "-pde",
        "--path_description_encoding",
        type=Path,
        required=True,
        help="Path from which encoded descriptions of themes are loaded",
    )
    parser.add_argument(
        "-psil",
        "--path_sub_industry_label",
        type=Path,
        required=True,
        help="Path to which sub industry membership labels of themes are saved",
    )
    parser.add_argument(
        "-pil",
        "--path_industry_label",
        type=Path,
        required=True,
        help="Path to which industry membership labels of ordered "
        "sub industry labels are saved",
    )

    args = parser.parse_args()

    cluster_for_sub_and_industries(
        path_theme_encoding=args.path_theme_encoding,
        path_description_encoding=args.path_description_encoding,
        path_sub_industry_label=args.path_sub_industry_label,
        path_industry_label=args.path_industry_label,
    )
