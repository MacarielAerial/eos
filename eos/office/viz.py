"""
Generates a static website for kedro pipelines
"""

import json
from pathlib import Path
import logging
import hashlib
from collections import defaultdict
from typing import Dict, Any, Set, List, Union
from toposort import toposort_flatten
import webbrowser
from functools import partial
import http.server
import socketserver
import yaml

from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from pipelinex import FlexiblePipeline, HatchDict
from flask import Flask, abort, jsonify, send_from_directory

from eos.utils import get_feed_dict

log = logging.getLogger(__name__)
_DATA = None  # type: Dict
_CATALOG = None  # type: DataCatalog
_JSON_NODES = {}
_DEFAULT_KEY = "__default__"
app = Flask(  # pylint: disable=invalid-name
    __name__, static_folder=str(Path(__file__).parent.absolute() / "html" / "static")
)


@app.route("/")
@app.route("/<path:subpath>")
def root(subpath="index.html"):
    """Serve the non static html and js etc"""
    return send_from_directory(
        str(Path(__file__).parent.absolute() / "html"), subpath, cache_timeout=0
    )

def call_viz(catalog: DataCatalog, pipelines: Dict[str, FlexiblePipeline]) -> None:
    global _DATA
    global _CATALOG
    _CATALOG = catalog
    conf_loader: ConfigLoader = ConfigLoader(
        conf_paths=["eos/conf/base", "eos/conf/local"]
    )
    conf_pipeline: Dict[str, Any] = conf_loader.get("pipelines*", "pipelines*/**")
    ae_pipeline: FlexiblePipeline = HatchDict(conf_pipeline).get("autoencoder_pipeline")
    nx_pipeline: FlexiblePipeline = HatchDict(conf_pipeline).get("networkx_pipeline")
    dgl_pipeline: FlexiblePipeline = HatchDict(conf_pipeline).get("dgl_pipeline")

    pipelines: Dict[str, FlexiblePipeline] = {"autoencoder_pipeline": ae_pipeline,
                                              "networkx_pipeline": nx_pipeline,
                                              "dgl_pipeline": dgl_pipeline,
                                              "master_pipeline": ae_pipeline + nx_pipeline + dgl_pipeline}
    _DATA = format_pipelines_data(pipelines = pipelines, catalog = catalog)
    Path("./public/pipeline.json").write_text(json.dumps(_DATA, indent=4, sort_keys=True))

    browser = True
    port = 4141
    host = "127.0.0.1"
    is_localhost = host in ("127.0.0.1", "localhost", "0.0.0.0")
    if browser and is_localhost:
        webbrowser.open_new("http://{}:{:d}/".format(host, port))
    app.run(host=host, port=port)

def run_static_server(directory: Union[str, Path] = "./public", port: int = 4141) -> None:
    """Serves content from the given directory on the given port
    FOR DEVELOPMENT USE ONLY, use a real server for production.
    behaves very much like `python -m http.server`
    Arguments:
        directory {[str]} -- Path to the directory to serve.
        port {[int]} -- TCP port that viz will listen to
    """
    here = Path(directory).absolute()
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(here))
    with socketserver.TCPServer(("", port), handler) as httpd:
        print("kedro-static-viz serving at port", port)
        httpd.serve_forever()

def format_pipelines_data(pipelines: Dict[str, "Pipeline"], catalog: DataCatalog) -> Dict[str, list]:
    """
    Format pipelines and catalog data from Kedro for kedro-viz.
    Args:
        pipelines: Dictionary of Kedro pipeline objects.
    Returns:
        Dictionary of pipelines, nodes, edges, tags and layers, and pipelines list.
    """
    pipelines_list = []
    # keep track of a sorted list of nodes to returned to the client
    nodes_list = []
    # keep track of edges in the graph: [{source_node_id -> target_node_id}]
    edges_list = []
    # keep tracking of node_id -> node data in the graph
    nodes = {}
    # keep track of node_id -> set(child_node_ids) for layers sorting
    node_dependencies = defaultdict(set)
    tags = set()

    for pipeline_key, pipeline in pipelines.items():
        pipelines_list.append({"id": pipeline_key, "name": _pretty_name(pipeline_key)})
        format_pipeline_data(
            catalog,
            pipeline_key,
            pipeline,
            nodes,
            node_dependencies,
            tags,
            edges_list,
            nodes_list,
        )

    # sort tags
    sorted_tags = [{"id": tag, "name": _pretty_name(tag)} for tag in sorted(tags)]
    # sort layers
    sorted_layers = _sort_layers(nodes, node_dependencies)

    default_pipeline = {"id": _DEFAULT_KEY, "name": _pretty_name(_DEFAULT_KEY)}
    selected_pipeline = (
        default_pipeline["id"]
        if default_pipeline in pipelines_list
        else pipelines_list[0]["id"]
    )

    return {
        "nodes": nodes_list,
        "edges": edges_list,
        "tags": sorted_tags,
        "layers": sorted_layers,
        "pipelines": pipelines_list,
        "selected_pipeline": selected_pipeline,
    }

def _construct_layer_mapping(catalog: DataCatalog) -> dict:
    if catalog.layers is None:
        return {ds_name: None for ds_name in catalog._data_sets}

    dataset_to_layer = {}
    for layer, dataset_names in catalog.layers.items():
        dataset_to_layer.update({dataset_name: layer for dataset_name in dataset_names})

    return dataset_to_layer


def _pretty_name(name: str) -> str:
    name = name.replace("-", " ").replace("_", " ")
    parts = [n.capitalize() for n in name.split()]
    return " ".join(parts)

def format_pipeline_data(
    catalog: DataCatalog,
    pipeline_key: str,
    pipeline: "Pipeline",  # noqa: F821
    nodes: Dict[str, dict],
    node_dependencies: Dict[str, Set[str]],
    tags: Set[str],
    edges_list: List[dict],
    nodes_list: List[dict],
) -> None:
    """Format pipeline and catalog data from Kedro for kedro-viz.
    Args:
        pipeline_key: key value of a pipeline object (e.g "__default__").
        pipeline: Kedro pipeline object.
        nodes: Dictionary of id and node dict.
        node_dependencies: Dictionary of id and node dependencies.
        edges_list: List of all edges.
        nodes_list: List of all nodes.
    """
    # keep_track of {data_set_namespace -> set(tags)}
    namespace_tags = defaultdict(set)
    # keep track of {data_set_namespace -> layer it belongs to}
    namespace_to_layer = {}

    dataset_to_layer = _construct_layer_mapping(catalog)

    # Nodes and edges
    for node in sorted(pipeline.nodes, key=lambda n: n.name):
        task_id = _hash(str(node))
        tags.update(node.tags)
        _JSON_NODES[task_id] = {"type": "task", "obj": node}
        if task_id not in nodes:
            nodes[task_id] = {
                "type": "task",
                "id": task_id,
                "name": getattr(node, "short_name", node.name),
                "full_name": getattr(node, "_func_name", str(node)),
                "tags": sorted(node.tags),
                "pipelines": [pipeline_key],
            }
            nodes_list.append(nodes[task_id])
        else:
            nodes[task_id]["pipelines"].append(pipeline_key)

        for data_set in node.inputs:
            namespace = data_set.split("@")[0]
            namespace_to_layer[namespace] = dataset_to_layer.get(data_set)
            namespace_id = _hash(namespace)
            edge = {"source": namespace_id, "target": task_id}
            if edge not in edges_list:
                edges_list.append(edge)
            namespace_tags[namespace].update(node.tags)
            node_dependencies[namespace_id].add(task_id)

            # if it is a parameter, add it to the node's data
            if _is_namespace_param(namespace):
                if "parameters" not in _JSON_NODES[task_id]:
                    _JSON_NODES[task_id]["parameters"] = {}

                if namespace == "parameters":
                    _JSON_NODES[task_id]["parameters"] = _get_dataset_data_params(
                        namespace
                    ).load()
                else:
                    parameter_name = namespace.replace("params:", "")
                    parameter_value = _get_dataset_data_params(namespace, catalog).load()
                    _JSON_NODES[task_id]["parameters"][parameter_name] = parameter_value

        for data_set in node.outputs:
            namespace = data_set.split("@")[0]
            namespace_to_layer[namespace] = dataset_to_layer.get(data_set)
            namespace_id = _hash(namespace)
            edge = {"source": task_id, "target": namespace_id}
            if edge not in edges_list:
                edges_list.append(edge)
            namespace_tags[namespace].update(node.tags)
            node_dependencies[task_id].add(namespace_id)
    # Parameters and data
    for namespace, tag_names in sorted(namespace_tags.items()):
        is_param = _is_namespace_param(namespace)
        node_id = _hash(namespace)

        _JSON_NODES[node_id] = {
            "type": "parameters" if is_param else "data",
            "obj": _get_dataset_data_params(namespace, catalog),
        }
        if is_param and namespace != "parameters":
            # Add "parameter_name" key only for "params:" prefix.
            _JSON_NODES[node_id]["parameter_name"] = namespace.replace("params:", "")

        if node_id not in nodes:
            nodes[node_id] = {
                "type": "parameters" if is_param else "data",
                "id": node_id,
                "name": _pretty_name(namespace),
                "full_name": namespace,
                "tags": sorted(tag_names),
                "layer": namespace_to_layer[namespace],
                "pipelines": [pipeline_key],
            }
            nodes_list.append(nodes[node_id])
        else:
            nodes[node_id]["pipelines"].append(pipeline_key)

def _hash(value):
    return hashlib.sha1(value.encode("UTF-8")).hexdigest()[:8]

def _is_namespace_param(namespace: str) -> bool:
    """Returns whether a dataset namespace is a parameter"""
    return namespace.lower().startswith("param")

def _get_dataset_data_params(namespace: str, catalog: DataCatalog):
    try:
        node_data = catalog._get_dataset(namespace)
    except DataSetNotFoundError:
        node_data = None
    return node_data

def _sort_layers(
    nodes: Dict[str, Dict], dependencies: Dict[str, Set[str]]
) -> List[str]:
    """Given a DAG represented by a dictionary of nodes, some of which have a `layer` attribute,
    along with their dependencies, return the list of all layers sorted according to
    the nodes' topological order, i.e. a layer should appear before another layer in the list
    if its node is a dependency of the other layer's node, directly or indirectly.
    For example, given the following graph:
        node1(layer=a) -> node2 -> node4 -> node6(layer=d)
                            |                   ^
                            v                   |
                          node3(layer=b) -> node5(layer=c)
    The layers ordering should be: [a, b, c, d]
    In theory, this is a problem of finding the
    [transitive closure](https://en.wikipedia.org/wiki/Transitive_closure) in a graph of layers
    and then toposort them. The algorithm below follows a repeated depth-first search approach:
        * For every node, find all layers that depends on it in a depth-first search.
        * While traversing, build up a dictionary of {node_id -> layers} for the node
        that has already been visited.
        * Turn the final {node_id -> layers} into a {layer -> layers} to represent the layers'
        dependencies. Note: the key is a layer and the values are the parents of that layer,
        just because that's the format toposort requires.
        * Feed this layers dictionary to ``toposort`` and return the sorted values.
        * Raise CircularDependencyError if the layers cannot be sorted topologically,
        i.e. there are cycles among the layers.
    Args:
        nodes: A dictionary of {node_id -> node} represents the nodes in the graph.
            A node's schema is:
                {
                    "type": str,
                    "id": str,
                    "name": str,
                    "layer": Optional[str]
                    ...
                }
        dependencies: A dictionary of {node_id -> set(child_ids)}
            represents the direct dependencies between nodes in the graph.
    Returns:
        The list of layers sorted based on topological order.
    Raises:
        CircularDependencyError: When the layers have cyclic dependencies.
    """
    node_layers = {}  # map node_id to the layers that depend on it

    def find_child_layers(node_id: str) -> Set[str]:
        """For the given node_id, find all layers that depend on it in a depth-first manner.
        Build up the node_layers dependency dictionary while traversing so each node is visited
        only once.
        Note: Python's default recursive depth limit is 1000, which means this algorithm won't
        work for pipeline with more than 1000 nodes. However, we can rewrite this using stack if
        we run into this limit in practice.
        """
        if node_id in node_layers:
            return node_layers[node_id]

        node_layers[node_id] = set()

        # for each child node of the given node_id,
        # mark its layer and all layers that depend on it as child layers of the given node_id.
        for child_node_id in dependencies[node_id]:
            child_node = nodes[child_node_id]
            child_layer = child_node.get("layer")
            if child_layer is not None:
                node_layers[node_id].add(child_layer)
            node_layers[node_id].update(find_child_layers(child_node_id))

        return node_layers[node_id]

    # populate node_layers dependencies
    for node_id in nodes:
        find_child_layers(node_id)

    # compute the layer dependencies dictionary based on the node_layers dependencies,
    # represented as {layer -> set(parent_layers)}
    layer_dependencies = defaultdict(set)
    for node_id, child_layers in node_layers.items():
        node_layer = nodes[node_id].get("layer")

        # add the node's layer as a parent layer for all child layers.
        # Even if a child layer is the same as the node's layer, i.e. a layer is marked
        # as its own parent, toposort still works so we don't need to check for that explicitly.
        if node_layer is not None:
            for layer in child_layers:
                layer_dependencies[layer].add(node_layer)

    # toposort the layer_dependencies to find the layer order.
    # Note that for string, toposort_flatten will default to alphabetical order for tie-break.
    return toposort_flatten(layer_dependencies)
