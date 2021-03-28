"""
Defines AutoEncoder model class
"""

from typing import Any

from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder
import torch
from torch import nn
from torch.utils.data import Dataset


class AutoEncoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features) -> Any:
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed


class TorchDataset(Dataset):
    def __init__(self, data: ndarray) -> None:
        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> Any:
        X = self.data[idx]
        return X

class PreprocessEncoder:
    def __init__(self, df_input: DataFrame) -> None:
        print(f"PreprocessEncoder: Initiated with dataframe of shape {df_input.shape}")
        self.df_input = df_input

        self.df_output = df_input.copy()

    def ordinal_encode(self) -> None:
        self.dict_categories: dict = {}
        for cat_feat in self.df_input.attrs["cat_feats"]:
            array: ndarray = self.df_input[cat_feat].to_numpy().reshape(1, -1)
            encoder: OrdinalEncoder = OrdinalEncoder()
            encoded_array: ndarray = encoder.fit_transform(array)
            self.df_output[cat_feat] = encoded_array.flatten()
            categories = encoder.categories_
            self.dict_categories.update({cat_feat: categories})
        print(f"PreprocessEncoder: Ordinally encoded {len(self.dict_categories)} features")
