import pandas as pd
import numpy as np
from typing import *


def SMA(data: pd.DataFrame, 
        period: Union[int, List[int]], 
        column: str = "close") -> pd.DataFrame:
    if type(period) is int:
        period = [period]
    for p in period:
        data[f"SMA{p}"] = data[column].rolling(20).mean()
    return data

def std_bands(data: pd.DataFrame, 
              num_stds: int = 2, 
              column: str = "close") -> pd.DataFrame:
        
    data["std20"] = data[column].rolling(20).std()
    cols2drop = ["std20"]
    if not "SMA20" in data.columns:
        data = SMA(data=data, period=20, column=column)
        cols2drop.append("SMA20")

    data["BBUpper"] = data.SMA20 + (data.std20 * num_stds)
    data["BBLower"] = data.SMA20 - (data.std20 * num_stds)

    data = data.drop(columns=cols2drop)
    return data


