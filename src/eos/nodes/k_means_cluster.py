import logging
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def cluster_encoding(encoding: np.ndarray) -> np.ndarray:
    # Identify the number of clusters
    # TODO: Identify a better strategy to tune the hyperparameter grid
    n_lowest = encoding.shape[0] // 5
    n_highest = encoding.shape[0] // 2
    silhouette_scores: List[Tuple[int, float]] = []
    for n_clusters in range(n_lowest, n_highest):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(encoding)
        score = float(silhouette_score(encoding, kmeans.labels_))
        silhouette_scores.append((n_clusters, score))

    optimal_n_clusters = sorted(silhouette_scores, key=lambda x: x[1], reverse=True)[0][
        0
    ]

    # Cluster with optimised hyperparameter
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42).fit(encoding)
    labels = kmeans.labels_

    return labels  # type: ignore[no-any-return]


def derive_sub_industry_encoding(
    per_theme_encoding: np.ndarray, sub_industry_label: np.ndarray
) -> np.ndarray:
    # Identify unique labels which are sorted by default
    unique_labels = np.unique(sub_industry_label)
    as_list_sub_industry_encoding = []

    for label in unique_labels:
        # Filter encodings by sub industry
        grouped_theme_encoding = per_theme_encoding[sub_industry_label == label]

        # Aggregate encodings within the group (mean pooling for now)
        group_encoding = np.mean(grouped_theme_encoding, axis=0)

        # Append sub industry encoding
        as_list_sub_industry_encoding.append(group_encoding)

    # Use ndarray instead of list
    sub_industry_encoding = np.array(as_list_sub_industry_encoding)

    return sub_industry_encoding


def _cluster_for_sub_and_industries(
    theme_encoding: np.ndarray, description_encoding: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # Average encodings as a simple baseline
    # Sector encoding is ignored if all input data is of the same sector
    # TODO: Experiment with transformer or GCN based embedding aggregation
    per_theme_encoding = (theme_encoding + description_encoding) / 2

    # Cluster for sub industry level
    sub_industry_label = cluster_encoding(per_theme_encoding)

    logger.info(
        f"Clustering results in {len(np.unique(sub_industry_label))} "
        "sub industry labels"
    )

    # Aggregate per theme encoding to obtain per sub industry encoding
    per_sub_industry_encoding = derive_sub_industry_encoding(
        per_theme_encoding=per_theme_encoding, sub_industry_label=sub_industry_label
    )

    # Cluster for industry level
    industry_label = cluster_encoding(per_sub_industry_encoding)

    logger.info(
        f"Clustering results in {len(np.unique(industry_label))} " "industry labels"
    )

    return sub_industry_label, industry_label
