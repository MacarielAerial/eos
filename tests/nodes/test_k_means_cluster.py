import numpy as np

from eos.nodes.k_means_cluster import (
    _cluster_for_sub_and_industries,
    cluster_encoding,
    derive_sub_industry_encoding,
)


def test_cluster_encoding() -> None:
    # Arrange
    encoding = np.random.rand(100, 5)  # 100 entities, 5-dimensional embedding

    # Act
    labels = cluster_encoding(encoding)

    # Assert
    assert isinstance(labels, np.ndarray), "Output should be a numpy ndarray."
    assert labels.shape == (100,), "Output shape should match the number of entities."
    assert (
        len(np.unique(labels)) <= 50
    ), "Number of clusters should not exceed half the number of entities."


def test_derive_sub_industry_encoding() -> None:
    # Arrange
    per_theme_encoding = np.random.rand(100, 5)
    sub_industry_label = np.random.randint(0, 10, size=100)

    # Act
    sub_industry_encoding = derive_sub_industry_encoding(
        per_theme_encoding, sub_industry_label
    )

    # Assert
    assert isinstance(
        sub_industry_encoding, np.ndarray
    ), "Output should be a numpy ndarray."
    assert len(sub_industry_encoding.shape) == 2, "Output should be 2-dimensional."
    assert (
        sub_industry_encoding.shape[1] == 5
    ), "Dimensionality of embeddings should be preserved."


def test_cluster_sub_and_industries() -> None:
    # Arrange
    theme_encoding = np.random.rand(100, 5)
    description_encoding = np.random.rand(100, 5)

    # Act
    sub_industry_label, industry_label = _cluster_for_sub_and_industries(
        theme_encoding, description_encoding
    )

    # Assert
    assert isinstance(
        sub_industry_label, np.ndarray
    ), "sub_industry_label should be a numpy ndarray."
    assert sub_industry_label.shape == (
        100,
    ), "sub_industry_label should have the correct shape."

    assert isinstance(
        industry_label, np.ndarray
    ), "industry_label should be a numpy ndarray."
    assert (
        industry_label.shape[0] < 100
    ), "industry_label should have the correct shape."
