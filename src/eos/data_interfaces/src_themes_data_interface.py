from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import dacite
import orjson

logger = logging.getLogger(__name__)


@dataclass
class SourceTheme:
    theme: str
    sector: str
    description: str


@dataclass
class SourceThemes:
    members: List[SourceTheme]

    @classmethod
    def from_untyped_jsonl(cls, filepath: Path) -> SourceThemes:
        members: List[SourceTheme] = []

        with open(filepath, "rb") as f:
            list_json_str = f.read().splitlines()
            for json_str in list_json_str:
                json_data = orjson.loads(json_str)

                src_theme = SourceTheme(**json_data)
                members.append(src_theme)

            source_themes = cls(members=members)

            logger.info(
                f"Loaded a {source_themes.__class__.__name__} object "
                f"from source jsonl {filepath}"
            )

            return source_themes


class SourceThemesDataInterface:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def save(self, source_themes: SourceThemes) -> None:
        with open(self.filepath, "wb") as f:
            json_str = orjson.dumps(source_themes, option=orjson.OPT_INDENT_2)
            f.write(json_str)

            logger.info(f"Saved a {type(source_themes)} object to {self.filepath}")

    def load(self) -> SourceThemes:
        with open(self.filepath, "rb") as f:
            json_data = orjson.loads(f.read())
            source_themes = dacite.from_dict(data_class=SourceThemes, data=json_data)

            logger.info(f"Loaded a {type(source_themes)} object from {self.filepath}")

            return source_themes
