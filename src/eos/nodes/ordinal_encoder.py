import logging

from pandas import DataFrame
from sklearn.preprocessing import _encoders

log = logging.getLogger(__name__)


def fit_ordinal_encoder(df: DataFrame) -> _encoders.OrdinalEncoder:
    """
    Fits an ordinal encoder to a dataframe
    """
    pass
