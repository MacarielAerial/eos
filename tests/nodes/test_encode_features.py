from unittest.mock import Mock

import numpy as np
from pytest import fixture
from sentence_transformers import SentenceTransformer

from eos.data_interfaces.node_dfs_data_interface import NodeAttrKey, NodeDFs
from eos.nodes.encode_features import _encode_features


@fixture
def mock_sentence_transformer() -> (  # type: ignore[no-any-unimported]
    SentenceTransformer
):
    mock_model = Mock()
    mock_model.encode = Mock(
        side_effect=lambda sentences, show_progress_bar: np.array(range(len(sentences)))
    )

    return mock_model


def test_encode_features(  # type: ignore[no-any-unimported]
    mock_sentence_transformer: SentenceTransformer, mock_node_dfs: NodeDFs
) -> None:
    expected_keys = [NodeAttrKey.theme, NodeAttrKey.description, NodeAttrKey.sector]
    expected_lengths = [2, 2, 1]

    results = list(_encode_features(mock_sentence_transformer, mock_node_dfs))

    assert len(results) == len(
        expected_keys
    ), "Number of FeatureEncodings does not match expected"

    for i, feature_encoding in enumerate(results):
        assert (
            feature_encoding.attr_key == expected_keys[i]
        ), f"Attribute key mismatch at index {i}"
        assert isinstance(
            feature_encoding.encoding, np.ndarray
        ), "Encoding is not an np.ndarray"
        assert (
            len(feature_encoding.encoding) == expected_lengths[i]
        ), f"Encoding length mismatch at index {i}"
