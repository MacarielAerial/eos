import logging
import shutil
from pathlib import Path

from pandas import DataFrame
from pytest import ExitCode, Session, fixture

from eos.data_interfaces.edge_dfs_data_interface import EdgeDF, EdgeDFs, EdgeType
from eos.data_interfaces.node_dfs_data_interface import NodeDF, NodeDFs, NodeType
from eos.data_interfaces.src_themes_data_interface import SourceTheme, SourceThemes

logger = logging.getLogger(__name__)


class TestDataPaths:
    @property
    def own_path(self) -> Path:
        return Path(__file__).parent

    @property
    def path_dir_data(self) -> Path:
        return self.own_path / "data"

    # Test input data paths

    @property
    def path_mock_untyped_jsonl(self) -> Path:
        return self.path_dir_data / "mock_industrial_business_theme_descriptions.jsonl"

    @property
    def path_mock_node_dfs(self) -> Path:
        return self.path_dir_data / "mock_node_dfs.json"

    @property
    def path_mock_edge_dfs(self) -> Path:
        return self.path_dir_data / "mock_edge_dfs.json"

    @property
    def path_mock_raw_source_themes(self) -> Path:
        return self.path_dir_data / "mock_raw_source_themes.jsonl"

    @property
    def path_mock_source_themes(self) -> Path:
        return self.path_dir_data / "mock_source_themes.json"

    # Test output data paths

    @property
    def path_dir_output(self) -> Path:
        return self.path_dir_data / "output"

    @property
    def path_saved_source_themes(self) -> Path:
        return self.path_dir_output / "saved_source_themes.jsonl"

    @property
    def path_saved_node_dfs(self) -> Path:
        return self.path_dir_output / "saved_node_dfs.json"

    @property
    def path_saved_edge_dfs(self) -> Path:
        return self.path_dir_output / "saved_edge_dfs.json"

    @property
    def path_parsed_source_themes(self) -> Path:
        return self.path_dir_output / "parsed_source_themes.json"

    @property
    def path_parsed_node_dfs(self) -> Path:
        return self.path_dir_output / "parsed_node_dfs.json"

    @property
    def path_parsed_edge_dfs(self) -> Path:
        return self.path_dir_output / "parsed_edge_dfs.json"


@fixture
def test_data_paths() -> TestDataPaths:
    return TestDataPaths()


@fixture
def mock_source_themes() -> SourceThemes:
    source_themes = SourceThemes(
        members=[
            SourceTheme(theme="efg", sector="abc", description="dwmdm w w dw."),
            SourceTheme(theme="hij", sector="abc", description="dsakj sd kjsd aj."),
        ]
    )

    return source_themes


@fixture
def mock_node_dfs() -> NodeDFs:
    node_dfs = NodeDFs(
        members=[
            NodeDF(ntype=NodeType.theme, df=DataFrame({"aaa": [1, 2], "bbb": [3, 4]})),
            NodeDF(ntype=NodeType.sector, df=DataFrame({"ccc": [5, 6], "ddd": [7, 8]})),
        ]
    )

    return node_dfs


@fixture
def mock_edge_dfs() -> EdgeDFs:
    edge_dfs = EdgeDFs(
        members=[
            EdgeDF(
                etype=EdgeType.theme_to_sector,
                df=DataFrame({"eee": [9, 10], "bbb": [11, 12]}),
            ),
        ]
    )

    return edge_dfs


def pytest_sessionstart(session: Session) -> None:
    path_dir_output = TestDataPaths().path_dir_output

    logger.info(
        f"A test data output directory at {path_dir_output} "
        "will be created if not exist already"
    )

    path_dir_output.mkdir(parents=True, exist_ok=True)


def pytest_sessionfinish(session: Session, exitstatus: ExitCode) -> None:
    path_dir_output = TestDataPaths().path_dir_output

    logger.info(f"Deleting Test output data directory at {path_dir_output}")

    shutil.rmtree(path=path_dir_output)
