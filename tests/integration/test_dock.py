"""
Tests data access functions in module "dock"
"""

from pipelinex import HatchDict
from pandas import DataFrame
from kedro.extras.datasets.pandas import CSVDataSet
import yaml

path_catalog_yml: str = "tests/data/catalog.yml"

def test_dock() -> None:
    input_key: str = "test_dock"
    catalog: dict = yaml.safe_load(open(path_catalog_yml, "r"))
    csv_dataset: CSVDataSet = HatchDict(catalog[input_key]).get("csv_dataset")
    csv_data: DataFrame = csv_dataset.load()
    assert isinstance(csv_data, DataFrame)
