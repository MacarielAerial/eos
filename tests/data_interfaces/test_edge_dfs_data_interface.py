from eos.data_interfaces.edge_dfs_data_interface import EdgeDFs, EdgeDFsDataInterface
from tests.conftest import TestDataPaths


def test_save(mock_edge_dfs: EdgeDFs, test_data_paths: TestDataPaths) -> None:
    edge_dfs_data_interface = EdgeDFsDataInterface(
        filepath=test_data_paths.path_saved_edge_dfs
    )
    edge_dfs_data_interface.save(mock_edge_dfs)

    assert test_data_paths.path_saved_edge_dfs.is_file()


def test_load(test_data_paths: TestDataPaths) -> None:
    edge_dfs_data_interface = EdgeDFsDataInterface(
        filepath=test_data_paths.path_mock_edge_dfs
    )
    edge_dfs = edge_dfs_data_interface.load()

    assert len(edge_dfs.members) == 1
