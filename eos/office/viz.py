# Functions in this module have been modified by the author of eos
#
# MIT License

# Copyright (c) 2020 Waylon Walker

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Generates a static website for kedro pipelines
"""

import http.server
import json
import logging
import socketserver
from functools import partial
from pathlib import Path
from typing import Dict, Union

from kedro.io import DataCatalog
from pipelinex import FlexiblePipeline

from eos.office.core import format_pipelines_data

log = logging.getLogger(__name__)


def call_viz(
    dir_static_site: str, catalog: DataCatalog, pipelines: Dict[str, FlexiblePipeline]
) -> None:
    """
    Creates a static web page with DAG visualisations and export to local

    Args:
        dir_static_site: Path to a local directory to store DAG information
        catalog: A catalog of all data structures used in pipelines
        pipelines: A name-pipeline dictionary of functions
    """
    # Parse structural info from FlexiblePipeline and DataCatalog objects
    _DATA = format_pipelines_data(pipelines=pipelines, catalog=catalog)

    # Export supplied pipeline information to local
    Path(dir_static_site).joinpath("pipeline.json").write_text(
        json.dumps(_DATA, indent=4, sort_keys=True)
    )

    # Run a Flask development server
    # port = 4141
    # host = "127.0.0.1"
    # app.run(host=host, port=port)


def run_static_server(directory: Union[str, Path], port: int = 4141) -> None:
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
