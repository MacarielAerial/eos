from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

import dacite
import orjson
from pandas import DataFrame

from eos.nodes.utils_df_serialisation import default, df_type_hook

logger = logging.getLogger(__name__)


class NodeAttrKey(str, Enum):
    nid = "nid"
    ntype = "ntype"
    theme = "theme"
    description = "description"


class NodeType(str, Enum):
    theme = "Theme"
    sector = "Sector"


@dataclass
class NodeDF:
    ntype: NodeType
    df: DataFrame


@dataclass
class NodeDFs:
    members: List[NodeDF]


class NodeDFsDataInterface:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def save(self, node_dfs: NodeDFs) -> None:
        with open(self.filepath, "wb") as f:
            json_str = orjson.dumps(
                node_dfs, default=default, option=orjson.OPT_INDENT_2
            )
            f.write(json_str)

            logger.info(f"Saved a {type(node_dfs)} type object to {self.filepath}")

    def load(self) -> NodeDFs:
        with open(self.filepath, "rb") as f:
            json_data = orjson.loads(f.read())
            node_dfs = dacite.from_dict(
                data_class=NodeDFs,
                data=json_data,
                config=dacite.Config(
                    type_hooks={DataFrame: df_type_hook}, cast=[NodeType]
                ),
            )

            logger.info(f"Loaded a {node_dfs} object from {self.filepath}")

            return node_dfs
