from eos.orchestrator import orchestrator

from networkx import Graph

conn_str: str = r"sqlite:///tests/data/e2e/user.db"
table_name = "user"


def test_e2e():
    G:Graph = orchestrator(conn_str = conn_str, table_name = table_name)
    assert isinstance(G, Graph)
