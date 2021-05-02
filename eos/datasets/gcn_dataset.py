"""
Defines data interface for IO operations on PyTorch Model model alone
"""

from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

import torch
from kedro.io import AbstractDataSet

from eos.classes.gcn import Model


class GCNDataSet(AbstractDataSet):
    def __init__(
        self,
        filepath: str,
        load_args: Optional[Dict[str, Any]] = None,
        save_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._filepath = PurePosixPath(filepath)
        self._load_args = load_args if load_args else {}
        self._save_args = save_args if save_args else {}

    def _load(self) -> Model:
        state_dict = torch.load(str(self._filepath))
        node_in_feats, node_out_feats, edge_in_feats, edge_hidden_feats = (
            state_dict["node_in_feats.weight"].shape[1],
            state_dict["node_out_feats.weight"].shape[1],
            state_dict["edge_in_feats.weight"].shape[1],
            state_dict["edge_hidden_feats.weight"].shape[1],
        )
        model = Model(node_in_feats, node_out_feats, edge_in_feats, edge_hidden_feats)
        model.load_state_dict(state_dict)
        return model

    def _save(self, model: Model) -> None:
        torch.save(model.state_dict(), str(self._filepath), **self._save_args)

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            load_args=self._load_args,
            save_args=self._save_args,
        )
