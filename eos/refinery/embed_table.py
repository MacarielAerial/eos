"""
Embeds categorical values of a dataframe
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
from pandas import DataFrame
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from eos.warehouse.autoencoder import AutoEncoder
from eos.warehouse.preprocess_encoder import PreprocessEncoder
from eos.warehouse.torch_dataset import TorchDataset

log = logging.getLogger(__name__)


def ordinally_encode_table(df: DataFrame) -> Tuple[DataFrame, Dict[str, List[str]]]:
    """
    Converts categories into integers
    """
    pe_obj: PreprocessEncoder = PreprocessEncoder(df_input=df)
    pe_obj.ordinal_encode()
    df_encoded = pe_obj.df_encoded
    categories = pe_obj.categories

    return df_encoded, categories


def df_to_dataloader(df: DataFrame) -> DataLoader:
    """
    Converts a dataframe into a dataloader through PyTorch's Dataset class
    """
    dataset: TorchDataset = TorchDataset(df=df)
    dataloader: DataLoader = DataLoader(dataset)

    return dataloader


def train_autoencoder(dataloader: DataLoader, params: Dict[str, Any]) -> AutoEncoder:
    """
    Trains an autoencoder model for categorical embedding
    """
    log.info(f"Loaded AutoEncoder model parameters: {params}")
    torch_dataset: TorchDataset = dataloader.dataset  # type: ignore
    input_shape: int = torch_dataset.arrays.shape[1]  # type: ignore
    model = AutoEncoder(input_shape=input_shape)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epochs = params["epochs"]

    # Train model
    for epoch in range(epochs):
        loss: float = 0.0
        for batch_features in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features.float())
            train_loss = criterion(outputs, batch_features.float())
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(dataloader)
        log.info("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    return model


def infer_with_autoencoder(df: DataFrame, autoencoder: AutoEncoder) -> DataFrame:
    """
    Infers numeric values for categorical features with trained AutoEncoder
    """
    if "cat_feats" not in df.attrs.keys():
        raise KeyError(
            f"Global variable space {df.attrs} does not contain cat_feats as a key"
        )
    indices_cat_feats: List[int] = [
        df[df.attrs["cat_feats"]].columns.get_loc(cat_feat)
        for cat_feat in df.attrs["cat_feats"]
    ]
    embedding = autoencoder(Tensor(df.to_numpy()))
    df[df.attrs["cat_feats"]] = embedding.detach().numpy()[:, indices_cat_feats]
    return df
