from eos.orchestrator import orchestrator
from os.path import realpath, isfile
from networkx import Graph

table_name: str = "user"
path_from_wd: str = "tests/data/e2e/user.db"

def check_db_exist(path):
    db_exist: bool = isfile(path)
    if not db_exist:
        raise FileNotFoundError(f"{path} does not exist")

def test_e2e():
    check_db_exist(path = realpath(path_from_wd))
    conn_str: str = "sqlite:///" + realpath(path_from_wd)
    G: Graph = orchestrator(conn_str = conn_str, table_name = table_name)
    assert isinstance(G, Graph)
