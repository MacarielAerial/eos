"""
Utility functions including IO operations
"""

from json import JSONEncoder
from typing import Any

import numpy as np


class NumpyEncoder(JSONEncoder):
    """
    Converts non-default data types into default data types before serialisation
    """

    def default(self, obj) -> Any:
        if hasattr(obj, "to_json"):
            return obj.to_json()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
