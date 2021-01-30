from eos.orchestrator import orchestrator
from os.path import realpath, isfile
from networkx import Graph

table_name: str = "user"
path_from_wd: str = "tests/data/e2e/user.db"

def check_db_exist(path):
    return isfile(path)

def test_e2e():
    if not check_db_exist(realpath(path_from_wd)):
        raise FileNotFoundError(f"{realpath(path_from_wd)} does not exist")
    conn_str: str = "sqlite:///" + realpath(path_from_wd)
    G: Graph = orchestrator(conn_str = conn_str, table_name = table_name)
    assert isinstance(G, Graph)
