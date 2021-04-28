"""
Defines data interface for IO operations on AutoEncoder model alone
"""

from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

import torch
from kedro.io import AbstractDataSet

from eos.classes.autoencoder import AutoEncoder


class AutoEncoderDataSet(AbstractDataSet):
    def __init__(
        self,
        filepath: str,
        load_args: Optional[Dict[str, Any]] = None,
        save_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._filepath = PurePosixPath(filepath)
        self._load_args = load_args if load_args else {}
        self._save_args = save_args if save_args else {}

    def _load(self) -> AutoEncoder:
        state_dict = torch.load(str(self._filepath))
        input_shape = state_dict["encoder_hidden_layer.weight"].shape[1]
        autoencoder = AutoEncoder(input_shape=input_shape)
        autoencoder.load_state_dict(state_dict)
        return autoencoder

    def _save(self, autoencoder: AutoEncoder) -> None:
        torch.save(autoencoder.state_dict(), str(self._filepath), **self._save_args)

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            load_args=self._load_args,
            save_args=self._save_args,
        )
