import tempfile
from pathlib import Path

import orjson
from pytest import fixture

from eos.data_interfaces.src_themes_data_interface import (
    SourceTheme,
    SourceThemes,
    SourceThemesDataInterface,
)
from tests.conftest import TestDataPaths


@fixture
def mock_source_themes() -> SourceThemes:
    source_themes = SourceThemes(
        members=[
            SourceTheme(theme="efg", sector="abc", description="dwmdm w w dw."),
            SourceTheme(theme="hij", sector="abc", description="dsakj sd kjsd aj."),
        ]
    )

    return source_themes


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
