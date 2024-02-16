import logging
from dataclasses import dataclass
from typing import Generator, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from eos.data_interfaces.edge_dfs_data_interface import EdgeAttrKey
from eos.data_interfaces.node_dfs_data_interface import NodeAttrKey, NodeDFs, NodeType

logger = logging.getLogger(__name__)


@dataclass
class FeatureEncoding:
    attr_key: NodeAttrKey | EdgeAttrKey
    encoding: np.ndarray


def encode_list_text(  # type: ignore[no-any-unimported]
    model: SentenceTransformer,
    list_text: List[str],
) -> np.ndarray:
    return model.encode(sentences=list_text, show_progress_bar=True)  # type: ignore[no-any-return]


def _encode_features(  # type: ignore[no-any-unimported]
    model: SentenceTransformer, node_dfs: NodeDFs
) -> Generator[FeatureEncoding, None, None]:
    # Index feature dataframes by types
    ntype_to_df = node_dfs.to_dict()

    # Gather input raw features
    raw_features: List[Tuple[NodeAttrKey | EdgeAttrKey, List[str]]] = [
        (
            NodeAttrKey.theme,
            ntype_to_df[NodeType.theme][NodeAttrKey.theme.value].tolist(),
        ),
        (
            NodeAttrKey.description,
            ntype_to_df[NodeType.theme][NodeAttrKey.description.value].tolist(),
        ),
        (
            NodeAttrKey.sector,
            ntype_to_df[NodeType.sector][NodeAttrKey.sector.value].tolist(),
        ),
    ]

    # Yield encoding for one feature at a time to optimise memory usage
    for attr_key, list_text in raw_features:
        logger.info(f"Encoding feature {attr_key} of length {len(list_text)}...")
        yield FeatureEncoding(
            attr_key=attr_key,
            encoding=encode_list_text(model=model, list_text=list_text),
        )
