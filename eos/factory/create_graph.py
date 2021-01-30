import networkx as nx
import numpy as np
import pandas as pd
from networkx import Graph
from pandas import DataFrame


class GraphCreator(object):
    """
    Create a graph from a 2D array of tabular data
    """

    def __init__(self, df_input: DataFrame):
        self.df_input = df_input

    def df_to_graph(self) -> Graph:
        G: Graph = Graph()
        return G
