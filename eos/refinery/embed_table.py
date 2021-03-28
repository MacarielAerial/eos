"""
Embeds categorical values of a dataframe
"""

from typing import Dict, Any, Tuple
import logging

from pandas import DataFrame
from torch.utils.data import DataLoader
from torch import nn, optim

from eos.warehouse.preprocess_encoder import PreprocessEncoder
from eos.warehouse.torch_dataset import TorchDataset
from eos.warehouse.autoencoder import AutoEncoder

log = logging.getLogger(__name__)


def ordinally_encode_table(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Converts categories into integers
    """
    pe_obj: PreprocessEncoder = PreprocessEncoder(df_input = df)
    pe_obj.ordinal_encode()
    df_encoded = pe_obj.df_encoded
    categories = pe_obj.categories

    return df_encoded, categories

def df_to_dataloader(df: DataFrame) -> DataLoader:
    """
    Converts a dataframe into a dataloader through PyTorch's Dataset class
    """
    dataset: TorchDataset = TorchDataset(df = df)
    dataloader: DataLoader = DataLoader(dataset)

    return dataloader

def train_autoencoder(dataloader: DataLoader, params: Dict[str, Any]) -> AutoEncoder:
    """
    Trains an autoencoder model for categorical embedding
    """
    log.info(f"Loaded AutoEncoder model parameters: {params}")
    input_shape: int = dataloader.dataset.arrays.shape[1]
    model = AutoEncoder(input_shape = input_shape)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epochs = params["epochs"]

    # Train model
    for epoch in range(epochs):
        loss = 0
        for batch_features in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features.float())
            train_loss = criterion(outputs, batch_features.float())
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(dataloader)
        log.debug("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    return model
