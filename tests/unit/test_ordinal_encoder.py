from pandas import DataFrame

from eos.nodes.ordinal_encoder import fit_ordinal_encoder


def test_fit_ordinal_encoder() -> None:
    df = DataFrame({"col_1": ["a", "b", "c"], "col_2": ["d", "e", "f"]})
    enc = fit_ordinal_encoder(df=df, params={"cols_to_encode": ["col_1", "col_2"]})
    array_encoded = enc.transform(df)

    assert array_encoded[1, 0] == 1.0
