import logging
import shutil
from pathlib import Path

from pytest import ExitCode, Session, fixture

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
        return (
            Path(self.path_dir_data)
            / "mock_industrial_business_theme_descriptions.jsonl"
        )

    # Test output data paths

    @property
    def path_dir_output(self) -> Path:
        return self.path_dir_data / "output"

    @property
    def path_saved_source_themes(self) -> Path:
        return self.path_dir_output / "saved_source_themes.jsonl"


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
