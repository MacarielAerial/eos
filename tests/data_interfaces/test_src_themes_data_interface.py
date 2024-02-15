import tempfile
from pathlib import Path

import orjson

from eos.data_interfaces.src_themes_data_interface import (
    SourceThemes,
    SourceThemesDataInterface,
)
from tests.conftest import TestDataPaths


def test_from_untyped_jsonl(test_data_paths: TestDataPaths) -> None:
    source_themes = SourceThemes.from_untyped_jsonl(
        filepath=test_data_paths.path_mock_untyped_jsonl
    )

    assert len(source_themes.members) == 2


def test_save(test_data_paths: TestDataPaths, mock_source_themes: SourceThemes) -> None:
    source_themes_data_interface = SourceThemesDataInterface(
        filepath=test_data_paths.path_saved_source_themes
    )
    source_themes_data_interface.save(source_themes=mock_source_themes)

    assert test_data_paths.path_saved_source_themes.is_file()


def test_load(mock_source_themes: SourceThemes) -> None:
    with tempfile.NamedTemporaryFile(mode="wb") as f:
        f.write(orjson.dumps(mock_source_themes))
        f.flush()

        path_source_themes = Path(f.name)

        source_themes_data_interface = SourceThemesDataInterface(
            filepath=path_source_themes
        )
        source_themes = source_themes_data_interface.load()

        assert len(source_themes.members) == 2
