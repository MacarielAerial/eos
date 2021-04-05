"""
Contains the Flask microservice for a static page
"""

from pathlib import Path

from flask import Flask, send_from_directory

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
