from pathlib import Path

from eos.data_interfaces.clusters_eval_data_interface import ClustersEvalDataInterface
from eos.data_interfaces.node_dfs_data_interface import NodeDFsDataInterface, NodeType
from eos.nodes.augment_element_dfs_with_llm import _augment_element_dfs_with_llm


def augment_element_dfs_with_llm(
    path_base_node_dfs: Path,
    path_sub_industry_eval: Path,
    path_industry_eval: Path,
    path_llm_node_dfs: Path,
) -> None:
    # Data Acess - Input
    base_node_dfs_data_interface = NodeDFsDataInterface(filepath=path_base_node_dfs)
    node_dfs = base_node_dfs_data_interface.load()
    node_dfs.validate()
    i_sub_industry = node_dfs.ntypes.index(NodeType.sub_industry)
    i_industry = node_dfs.ntypes.index(NodeType.industry)

    sub_industry_eval_data_interface = ClustersEvalDataInterface(
        filepath=path_sub_industry_eval
    )
    sub_industry_eval = sub_industry_eval_data_interface.load()

    industry_eval_data_interface = ClustersEvalDataInterface(
        filepath=path_industry_eval
    )
    industry_eval = industry_eval_data_interface.load()

    # Task Processing
    (
        node_dfs.members[i_sub_industry],
        node_dfs.members[i_industry],
    ) = _augment_element_dfs_with_llm(
        sub_industry_node_df=node_dfs.members[i_sub_industry],
        industry_node_df=node_dfs.members[i_industry],
        df_sub_industry_eval=sub_industry_eval.to_df(),
        df_industry_eval=industry_eval.to_df(),
    )
    node_dfs.validate()

    # Data Access - Output
    llm_node_dfs_data_interface = NodeDFsDataInterface(filepath=path_llm_node_dfs)
    llm_node_dfs_data_interface.save(node_dfs=node_dfs)


if __name__ == "__main__":
    import argparse

    from eos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Augments node dataframes with LLM generated output mostly "
        "including cluster text descriptions and evaluation notes"
    )
    parser.add_argument(
        "-pbnd",
        "--path_base_node_dfs",
        type=Path,
        required=True,
        help="Path from which non-LLM augmented node dataframes are loaded",
    )
    parser.add_argument(
        "-psie",
        "--path_sub_industry_eval",
        type=Path,
        required=True,
        help="Path from which sub industry evaluation data is loaded",
    )
    parser.add_argument(
        "-pie",
        "--path_industry_eval",
        type=Path,
        required=True,
        help="Path from which industry evaluation data is loaded",
    )
    parser.add_argument(
        "-plnd",
        "--path_llm_node_dfs",
        type=Path,
        required=True,
        help="Path to which LLM augmented node dataframes are saved",
    )

    args = parser.parse_args()

    augment_element_dfs_with_llm(
        path_base_node_dfs=args.path_base_node_dfs,
        path_sub_industry_eval=args.path_sub_industry_eval,
        path_industry_eval=args.path_industry_eval,
        path_llm_node_dfs=args.path_llm_node_dfs,
    )
