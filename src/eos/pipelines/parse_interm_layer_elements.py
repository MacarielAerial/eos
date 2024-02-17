from pathlib import Path

import numpy as np

from eos.data_interfaces.edge_dfs_data_interface import EdgeDFsDataInterface
from eos.data_interfaces.node_dfs_data_interface import NodeDFsDataInterface
from eos.nodes.parse_interm_layer_elements import _parse_interm_layer_elements


def parse_interm_layer_elements(
    path_base_node_dfs: Path,
    path_base_edge_dfs: Path,
    path_sub_industry_label: Path,
    path_industry_label: Path,
    path_interm_node_dfs: Path,
    path_interm_edge_dfs: Path,
) -> None:
    # Data Access - Input
    base_node_dfs_data_interface = NodeDFsDataInterface(filepath=path_base_node_dfs)
    base_node_dfs = base_node_dfs_data_interface.load()
    base_node_dfs.validate()

    base_edge_dfs_data_interface = EdgeDFsDataInterface(filepath=path_base_edge_dfs)
    base_edge_dfs = base_edge_dfs_data_interface.load()
    base_edge_dfs.validate()

    sub_industry_label = np.load(path_sub_industry_label)
    industry_label = np.load(path_industry_label)

    # Taks Processing
    interm_node_dfs, interm_edge_dfs = _parse_interm_layer_elements(
        base_node_dfs=base_node_dfs,
        base_edge_dfs=base_edge_dfs,
        sub_industry_label=sub_industry_label,
        industry_label=industry_label,
    )

    # Data Access - Output
    interm_node_dfs_data_interface = NodeDFsDataInterface(filepath=path_interm_node_dfs)
    interm_node_dfs_data_interface.save(node_dfs=interm_node_dfs)

    interm_edge_dfs_data_interface = EdgeDFsDataInterface(filepath=path_interm_edge_dfs)
    interm_edge_dfs_data_interface.save(edge_dfs=interm_edge_dfs)


if __name__ == "__main__":
    import argparse

    from eos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Parses intermediate layer nodes and edges from base layer "
        "nodes and edges as well as clustering result arrays"
    )
    parser.add_argument(
        "-pbnd",
        "--path_base_node_dfs",
        type=Path,
        required=True,
        help="Path from which base layer node dataframes are loaded",
    )
    parser.add_argument(
        "-pbed",
        "--path_base_edge_dfs",
        type=Path,
        required=True,
        help="Path from which base layer edge dataframes are loaded",
    )
    parser.add_argument(
        "-psil",
        "--path_sub_industry_label",
        type=Path,
        required=True,
        help="Path from sub-industry clustering result is loaded",
    )
    parser.add_argument(
        "-pil",
        "--path_industry_label",
        type=Path,
        required=True,
        help="Path from industry clustering result is loaded",
    )
    parser.add_argument(
        "-pind",
        "--path_interm_node_dfs",
        type=Path,
        required=True,
        help="Path to which intermediate layer node dataframes are saved",
    )
    parser.add_argument(
        "-pied",
        "--path_interm_edge_dfs",
        type=Path,
        required=True,
        help="Path to which intermediate layer edge dataframes are saved",
    )

    args = parser.parse_args()

    parse_interm_layer_elements(
        path_base_node_dfs=args.path_base_node_dfs,
        path_base_edge_dfs=args.path_base_edge_dfs,
        path_sub_industry_label=args.path_sub_industry_label,
        path_industry_label=args.path_industry_label,
        path_interm_node_dfs=args.path_interm_node_dfs,
        path_interm_edge_dfs=args.path_interm_edge_dfs,
    )
