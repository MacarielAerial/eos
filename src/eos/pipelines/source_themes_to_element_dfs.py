from pathlib import Path

from eos.data_interfaces.edge_dfs_data_interface import EdgeDFsDataInterface
from eos.data_interfaces.node_dfs_data_interface import NodeDFsDataInterface
from eos.data_interfaces.src_themes_data_interface import SourceThemesDataInterface
from eos.nodes.source_themes_to_element_dfs import _source_themes_to_element_dfs


def source_themes_to_element_dfs(
    path_source_themes: Path, path_node_dfs: Path, path_edge_dfs: Path
) -> None:
    # Data Access - Input
    source_themes_data_interface = SourceThemesDataInterface(
        filepath=path_source_themes
    )
    source_themes = source_themes_data_interface.load()

    # Task Processing
    node_dfs, edge_dfs = _source_themes_to_element_dfs(source_themes=source_themes)

    # Data Access - Output
    node_dfs_data_interface = NodeDFsDataInterface(filepath=path_node_dfs)
    node_dfs_data_interface.save(node_dfs=node_dfs)

    edge_dfs_data_interface = EdgeDFsDataInterface(filepath=path_edge_dfs)
    edge_dfs_data_interface.save(edge_dfs=edge_dfs)


if __name__ == "__main__":
    import argparse

    from eos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Turns typed data into dataframes of graph elements"
    )
    parser.add_argument(
        "-pst",
        "--path_source_themes",
        type=Path,
        required=True,
        help="Path from which preliminarily typed data is loaded",
    )
    parser.add_argument(
        "-pnd",
        "--path_node_dfs",
        type=Path,
        required=True,
        help="Path to which node dataframes are saved",
    )
    parser.add_argument(
        "-ped",
        "--path_edge_dfs",
        type=Path,
        required=True,
        help="Path to which edge dataframes are saved",
    )

    args = parser.parse_args()

    source_themes_to_element_dfs(
        path_source_themes=args.path_source_themes,
        path_node_dfs=args.path_node_dfs,
        path_edge_dfs=args.path_edge_dfs,
    )
