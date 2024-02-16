from pathlib import Path

from eos.data_interfaces.src_themes_data_interface import (
    SourceThemes,
    SourceThemesDataInterface,
)


def type_raw_source_themes(
    path_raw_source_themes: Path, path_source_themes: Path
) -> None:
    # Data Access - Input & Task Processing
    source_themes = SourceThemes.from_untyped_jsonl(filepath=path_raw_source_themes)
    source_themes.validate()

    # Data Access - Output
    source_themes_data_interface = SourceThemesDataInterface(
        filepath=path_source_themes
    )
    source_themes_data_interface.save(source_themes=source_themes)


if __name__ == "__main__":
    import argparse

    from eos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Types raw source themes data into its serialised "
        "python dataclass representation"
    )
    parser.add_argument(
        "-prst",
        "--path_raw_source_themes",
        required=True,
        type=Path,
        help="Path from which raw source themes data is loaded",
    )
    parser.add_argument(
        "-pst",
        "--path_source_themes",
        required=True,
        type=Path,
        help="Path to which typed source themes are saved",
    )

    args = parser.parse_args()

    type_raw_source_themes(
        path_raw_source_themes=args.path_raw_source_themes,
        path_source_themes=args.path_source_themes,
    )
