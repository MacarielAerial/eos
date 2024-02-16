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


class EdgeAttrKey(str, Enum):
    eid = "eid"
    etype = "etype"


class EdgeType(str, Enum):
    theme_to_sector = "ThemeToSector"


@dataclass
class EdgeDF:
    etype: EdgeType
    df: DataFrame


@dataclass
class EdgeDFs:
    members: List[EdgeDF]


class EdgeDFsDataInterface:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def save(self, edge_dfs: EdgeDFs) -> None:
        with open(self.filepath, "wb") as f:
            json_str = orjson.dumps(
                edge_dfs, default=default, option=orjson.OPT_INDENT_2
            )
            f.write(json_str)

            logger.info(f"Saved a {type(edge_dfs)} type object to {self.filepath}")

    def load(self) -> EdgeDFs:
        with open(self.filepath, "rb") as f:
            json_data = orjson.loads(f.read())
            edge_dfs = dacite.from_dict(
                data_class=EdgeDFs,
                data=json_data,
                config=dacite.Config(
                    type_hooks={DataFrame: df_type_hook}, cast=[EdgeType]
                ),
            )

            logger.info(f"Loaded a {edge_dfs} object from {self.filepath}")

            return edge_dfs
