from eos.orchestrator import orchestrator
from os.path import realpath
from networkx import Graph

table_name: str = "user"
path_from_wd: str = "tests/data/e2e/user.db"


def test_e2e():
    conn_str: str = "sqlite:///" + realpath(path_from_wd)
    G: Graph = orchestrator(conn_str = conn_str, table_name = table_name)
    assert isinstance(G, Graph)
