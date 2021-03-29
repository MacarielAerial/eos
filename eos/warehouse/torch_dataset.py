"""
Includes PyTorch Dataset definition
Dataset is a PyTorch class that is the valid input for PyTorch's DataLoader
"""

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    def __init__(self, df: DataFrame) -> None:
        self.df = df
        self.arrays: ndarray = df.to_numpy()

    def __len__(self) -> int:
        return self.arrays.shape[0]

    def __getitem__(self, idx) -> ndarray:
        X = np.array(self.arrays[idx].astype(float))
        return X
