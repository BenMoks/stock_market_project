"""
io_utils.py

Input/output utilities for saving and loading stock market data.
Provides functions for exporting analysis results to CSV and JSON formats.
"""

from pathlib import Path
from typing import Union, Dict, Any
import pandas as pd
import json


def save_dataframe_to_csv(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
    """
    Save a DataFrame to a CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filepath : Union[str, Path]
        Output file path.
    **kwargs
        Additional arguments passed to pandas.DataFrame.to_csv()
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)


def save_series_to_csv(series: pd.Series, filepath: Union[str, Path], **kwargs) -> None:
    """
    Save a pandas Series to a CSV file.
    
    Parameters
    ----------
    series : pd.Series
        Series to save.
    filepath : Union[str, Path]
        Output file path.
    **kwargs
        Additional arguments passed to pandas.Series.to_csv()
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path, **kwargs)


def save_dict_to_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save a dictionary to a JSON file.
    
    Useful for saving summary statistics, configuration, or analysis results.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary to save.
    filepath : Union[str, Path]
        Output file path.
    indent : int, default 2
        JSON indentation level for pretty printing.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy/pandas types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_data = convert_to_serializable(data)
    
    with open(path, 'w') as f:
        json.dump(serializable_data, f, indent=indent, default=str)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary.
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Input file path.
    
    Returns
    -------
    Dict[str, Any]
        Loaded dictionary data.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)
