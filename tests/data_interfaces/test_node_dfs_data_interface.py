from eos.data_interfaces.node_dfs_data_interface import NodeDFs, NodeDFsDataInterface
from tests.conftest import TestDataPaths


def test_save(mock_node_dfs: NodeDFs, test_data_paths: TestDataPaths) -> None:
    node_dfs_data_interface = NodeDFsDataInterface(
        filepath=test_data_paths.path_saved_node_dfs
    )
    node_dfs_data_interface.save(mock_node_dfs)

    assert test_data_paths.path_saved_node_dfs.is_file()


def test_load(test_data_paths: TestDataPaths) -> None:
    node_dfs_data_interface = NodeDFsDataInterface(
        filepath=test_data_paths.path_mock_node_dfs
    )
    node_dfs = node_dfs_data_interface.load()

    assert len(node_dfs.members) == 2
