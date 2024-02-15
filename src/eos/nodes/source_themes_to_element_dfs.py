import logging
from typing import List, Tuple

from pandas import DataFrame

from eos.data_interfaces.edge_dfs_data_interface import (
    EdgeAttrKey,
    EdgeDF,
    EdgeDFs,
    EdgeType,
)
from eos.data_interfaces.node_dfs_data_interface import (
    NodeAttrKey,
    NodeDF,
    NodeDFs,
    NodeType,
)
from eos.data_interfaces.src_themes_data_interface import SourceThemes

logger = logging.getLogger(__name__)


def source_themes_to_element_dfs(
    source_themes: SourceThemes,
) -> Tuple[NodeDFs, EdgeDFs]:
    # Iterables for Theme node data
    nids_theme: List[int] = []
    ntypes_theme: List[str] = []
    themes: List[str] = []
    descriptions: List[str] = []

    # Iterables for Sector node data
    nids_sector: List[int] = []
    ntypes_sector: List[str] = []
    sectors: List[str] = []

    # Iterables for ThemeToSector edge data
    eids_tts: List[Tuple[int, int]] = []
    etype_tts: List[Tuple[int, int]] = []

    # Execute collection of graph elements
    curr_nid: int = 0
    for source_theme in source_themes.members:
        # Collect data for Theme nodes
        nid_theme = curr_nid
        curr_nid += 1

        nids_theme.append(nid_theme)
        ntypes_theme.append(NodeType.theme.value)
        themes.append(source_theme.theme)
        descriptions.append(source_theme.description)

        # Skip Sector node data collection if already exists
        if source_theme.sector not in sectors:
            # Collect data for Sector nodes
            nid_sector = curr_nid
            curr_nid += 1

            nids_sector.append(nid_sector)
            ntypes_sector.append(NodeType.sector.value)
            sectors.append(source_theme.sector)

        # Collect data for ThemeToSector edges
        i_sector = sectors.index(source_theme.sector)
        nid_sector = nids_sector[i_sector]

        eids_tts.append((nid_theme, nid_sector))
        etype_tts.append(EdgeType.theme_to_sector.value)

    # Compile DataFrame objects from iterables
    df_theme = DataFrame(
        {
            NodeAttrKey.nid.value: nids_theme,
            NodeAttrKey.ntype.value: ntypes_theme,
            NodeAttrKey.theme.value: themes,
        }
    )
    df_sector = DataFrame(
        {NodeAttrKey.nid.value: nids_sector, NodeAttrKey.ntype.value: ntypes_sector}
    )
    df_tts = DataFrame(
        {EdgeAttrKey.eid.value: eids_tts, EdgeAttrKey.etype.value: etype_tts}
    )

    logger.info(
        f"{NodeType.theme.value} node dataframe is shaped {df_theme.shape} "
        f"and has columns {df_theme.columns}"
    )
    logger.info(
        f"{NodeType.sector.value} node dataframe is shaped {df_sector.shape} "
        f"and has columns {df_sector.columns}"
    )
    logger.info(
        f"{EdgeType.theme_to_sector.value} edge dataframe is shaped {df_tts.shape} "
        f"and has columns {df_tts.columns}"
    )

    node_dfs = NodeDFs(
        members=[
            NodeDF(ntype=NodeType.theme, df=df_theme),
            NodeDF(ntype=NodeType.sector, df=df_sector),
        ]
    )
    edge_dfs = EdgeDFs(members=[EdgeDF(etype=EdgeType.theme_to_sector, df=df_tts)])

    return node_dfs, edge_dfs
