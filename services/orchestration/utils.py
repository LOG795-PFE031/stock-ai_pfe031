import numpy as np
import pandas as pd

from core.types import ProcessedData


def to_processed_data(d: dict) -> ProcessedData:
    """
    Convert a dictionary into a ProcessedData object, determining the appropriate
    format for the `X` attribute based on its content.

    The function handles different structures of input data:
    - If `X` is a list of dictionaries (List[Dict[str, Any]]), it is converted into a pandas
        DataFrame.
    - If `X` is a list of floats or list of lists (List[float] or List[List[float]]), it is
        converted into a NumPy array.
    - If `X` is not present or not in a supported format, it is set to None.

    Returns:
        ProcessedData: An object containing structured feature and target data, along with metadata.
    """
    x_raw = d.get("X")

    if isinstance(x_raw, list):
        if x_raw and isinstance(x_raw[0], dict):
            X = pd.DataFrame(x_raw)
        else:
            X = np.array(x_raw)
    else:
        X = None

    return ProcessedData(
        X=X,
        y=np.array(d.get("y")) if d.get("y") is not None else None,
        feature_index_map=d.get("feature_index_map", None),
        start_date=d.get("start_date", None),
        end_date=d.get("end_date", None),
    )
