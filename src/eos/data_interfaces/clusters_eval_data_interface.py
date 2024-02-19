from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, TypedDict

import dacite
import orjson

logger = logging.getLogger(__name__)


class ClusteringLevel(str, Enum):
    sub_industry = "sub_industry"
    industry = "industry"


class RawClusterDict(TypedDict):
    cluster_label: str
    evaluation_note: str


@dataclass
class ClusterEval:
    id: int  # "label" produced by clustering algorithms
    label: str  # LLM populates this attribute
    note: str  # LLM populates this attribute

    @classmethod
    def from_raw_dict(
        cls,
        cluster_id: str,  # string formatted integer because of json
        raw_dict: RawClusterDict,
    ) -> ClusterEval:
        return cls(
            id=int(cluster_id),
            label=raw_dict["cluster_label"],
            note=raw_dict["evaluation_note"],
        )


@dataclass
class ClustersEval:
    level: ClusteringLevel
    members: List[ClusterEval]

    @classmethod
    def from_raw_dict(
        cls,
        # TODO: Adjust LLM json output parsing process to infer level
        level: ClusteringLevel,  # Manual input
        raw_dict: Dict[str, RawClusterDict],
    ) -> ClustersEval:
        return cls(
            level=level,
            members=[
                ClusterEval.from_raw_dict(cluster_id=k, raw_dict=v)
                for k, v in raw_dict.items()
            ],
        )


class ClustersEvalDataInterface:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def save(self, clusters_eval: ClustersEval) -> None:
        if not self.filepath.parent.exists():
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Creating {self.filepath.parent} because it does not yet exist"
            )

        with open(self.filepath, "wb") as f:
            f.write(orjson.dumps(clusters_eval, option=orjson.OPT_INDENT_2))

            logger.info(f"Saved a {type(clusters_eval)} object to {self.filepath}")

    def load(self) -> ClustersEval:
        with open(self.filepath, "rb") as f:
            json_data = orjson.loads(f.read())

            clusters_eval = dacite.from_dict(
                data_class=ClustersEval,
                data=json_data,
                config=dacite.Config(cast=[ClusteringLevel]),
            )

            logger.info(f"Loaded a {type(clusters_eval)} object from {self.filepath}")

            return clusters_eval
