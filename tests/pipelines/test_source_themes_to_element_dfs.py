from eos.pipelines.source_themes_to_element_dfs import source_themes_to_element_dfs
from tests.conftest import TestDataPaths


def test_source_themes_to_element_dfs(test_data_paths: TestDataPaths) -> None:
    source_themes_to_element_dfs(
        path_source_themes=test_data_paths.path_mock_source_themes,
        path_node_dfs=test_data_paths.path_parsed_node_dfs,
        path_edge_dfs=test_data_paths.path_parsed_edge_dfs,
    )

    assert test_data_paths.path_parsed_node_dfs.is_file()
    assert test_data_paths.path_parsed_edge_dfs.is_file()
