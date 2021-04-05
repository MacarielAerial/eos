# Functions in this module have been modified by the author of eos
#
# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for DAG visualisation
"""

import hashlib
import inspect
from pathlib import Path
from typing import Any, Dict, Optional

from kedro.io import AbstractDataSet, DataCatalog, DataSetNotFoundError


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


def _hash(value):
    return hashlib.sha1(value.encode("UTF-8")).hexdigest()[:8]


def _is_namespace_param(namespace: str) -> bool:
    """Returns whether a dataset namespace is a parameter"""
    return namespace.lower().startswith("param")


def _get_dataset_data_params(namespace: str, catalog: DataCatalog):
    try:
        node_data: Optional[AbstractDataSet] = catalog._get_dataset(namespace)
    except DataSetNotFoundError:
        node_data = None
    return node_data


def _get_task_metadata(node):
    """Get a dictionary of task metadata: 'code', 'filepath' and 'docstring'.
    For 'filepath', remove the path to the project from the full code location
    before sending to JSON.
    Example:
        'code_full_path':   'path-to-project/project_root/path-to-code/node.py'
        'Path.cwd().parent':'path-to-project/'
        'filepath':    'project_root/path-to-code/node.py''
    """
    task_metadata = {"code": inspect.getsource(node["obj"]._func)}

    code_full_path = Path(inspect.getfile(node["obj"]._func)).expanduser().resolve()
    filepath = code_full_path.relative_to(Path.cwd().parent)
    task_metadata["filepath"] = str(filepath)

    docstring = inspect.getdoc(node["obj"]._func)
    if docstring:
        task_metadata["docstring"] = docstring

    if "parameters" in node:
        task_metadata["parameters"] = node["parameters"]

    return task_metadata


def _get_dataset_metadata(node):
    dataset = node["obj"]
    if dataset:
        dataset_metadata = {
            "type": f"{dataset.__class__.__module__}.{dataset.__class__.__qualname__}",
            "filepath": str(dataset._describe().get("filepath")),
        }
    else:
        # dataset not persisted, so no metadata defined in catalog.yml.
        dataset_metadata = {}
    return dataset_metadata


def _get_parameter_values(node: Dict) -> Any:
    """Get parameter values from a stored node."""
    if node["obj"] is not None:
        parameter_values = node["obj"].load()
    else:  # pragma: no cover
        parameter_values = {}
    return parameter_values
