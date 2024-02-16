from eos.pipelines.type_raw_source_themes import type_raw_source_themes
from tests.conftest import TestDataPaths


def test_type_raw_source_themes(test_data_paths: TestDataPaths) -> None:
    type_raw_source_themes(
        path_raw_source_themes=test_data_paths.path_mock_raw_source_themes,
        path_source_themes=test_data_paths.path_parsed_source_themes,
    )

    assert test_data_paths.path_parsed_source_themes.is_file()
