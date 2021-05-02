import logging
from typing import Any, Dict, List

from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder

log = logging.getLogger(__name__)


def fit_ordinal_encoder(df: DataFrame, params: Dict[str, Any]) -> OrdinalEncoder:
    """
    Fits an ordinal encoder to a dataframe
    """
    cols_to_encode: List[str] = params["cols_to_encode"]
    enc = OrdinalEncoder()
    log.info(f"Encoding the following columns:\n{cols_to_encode}")
    enc.fit(df[cols_to_encode])

    return enc


def transform_with_ordinal_encoder(
    df: DataFrame, enc: OrdinalEncoder, params: Dict[str, Any]
) -> DataFrame:
    """
    Transforms a dataframe with an ordinal encoder
    """
    cols_to_transform: List[str] = params["cols_to_transform"]

    log.info(
        f"Transforming the following columns with the ordinal encoder:\n{cols_to_transform}"
    )
    df[cols_to_transform] = enc.transform(df[cols_to_transform])

    return df
