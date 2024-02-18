import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from eos.data_interfaces.node_dfs_data_interface import NodeDFsDataInterface
from eos.nodes.encode_features import _encode_features

logger = logging.getLogger(__name__)


def encode_features(
    path_node_dfs: Path,
    path_sentence_transformer: Path,
    path_dir_feature_encoding: Path,
) -> None:
    # Data Access - Input
    node_dfs_data_interface = NodeDFsDataInterface(filepath=path_node_dfs)
    node_dfs = node_dfs_data_interface.load()

    model = SentenceTransformer(model_name_or_path=str(path_sentence_transformer))

    # Task Processing & Data Access - Output
    if not path_dir_feature_encoding.exists():
        logger.info(f"Creating {path_dir_feature_encoding} because it does not yet exist")
        path_dir_feature_encoding.mkdir(parents=True, exist_ok=True)

    for feature_encoding in _encode_features(model=model, node_dfs=node_dfs):
        path_feature_encoding = (
            path_dir_feature_encoding / feature_encoding.attr_key.value
        ).with_suffix(".npy")

        np.save(file=path_feature_encoding, arr=feature_encoding.encoding)

        logger.info(
            f"Saved encoding for feature {feature_encoding.attr_key} to {path_feature_encoding}"
        )


if __name__ == "__main__":
    import argparse

    from eos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Encodes text features with a sentence transformer model"
    )
    parser.add_argument(
        "-pnd",
        "--path_node_dfs",
        type=Path,
        required=True,
        help="Path from which node dataframes with raw text features are loaded",
    )
    parser.add_argument(
        "-pst",
        "--path_sentence_transformer",
        type=Path,
        required=True,
        help="Path to a directory from which a sentence transformer ideally optimised "
        "for semantic similarity is loaded",
    )
    parser.add_argument(
        "-pdfe",
        "--path_dir_feature_encoding",
        type=Path,
        required=True,
        help="Path to a directory into which feature encodings "
        "are saved one at a time",
    )

    args = parser.parse_args()

    encode_features(
        path_node_dfs=args.path_node_dfs,
        path_sentence_transformer=args.path_sentence_transformer,
        path_dir_feature_encoding=args.path_dir_feature_encoding,
    )
