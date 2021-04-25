from pandas import DataFrame

from eos.nodes.ordinal_encoder import fit_ordinal_encoder


def test_fit_ordinal_encoder() -> None:
    df = DataFrame()
    fit_ordinal_encoder(df=df)
