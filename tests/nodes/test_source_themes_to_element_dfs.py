from eos.data_interfaces.src_themes_data_interface import SourceThemes
from eos.nodes.source_themes_to_element_dfs import source_themes_to_element_dfs


def test_source_themes_to_element_dfs(mock_source_themes: SourceThemes) -> None:
    element_dfs = source_themes_to_element_dfs(source_themes=mock_source_themes)

    assert len(element_dfs[0].members) == 2
    assert len(element_dfs[1].members) == 1
