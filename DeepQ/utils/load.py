import re
from typing import *
import pandas as pd
import numpy as np
from os import listdir
from os.path import exists
from utils.price_utils import *


def load_file(path: str):
    ticker_name = re.split(r"[/\//]{1,}", path)[-1].replace(".csv", "")
    return {ticker_name: pd.read_csv(path)[["open", "high", "low", "close"]].values}

class DataLoader:
    """
    Loads Ticker Data given a parent directory
    """
    def __init__(self, 
                 data_dir: str,
                 sma_periods: List[int] = [],
                 bb_bands: bool = True):
                 
        self.dir = data_dir
        self.sma_periods = sma_periods
        self.bb_bands = bb_bands
        self.files = listdir(self.dir)

    def load_from_tickers(self, ticker_names: List[str]) -> np.ndarray:
        data = {}
        for ticker in ticker_names:
            ticker = ticker.replace(".csv", "")
            ticker_path = f"{self.dir}/{ticker}.csv"
            if self.__file_exists(ticker_path):
                data[ticker] = self.__load_file(ticker_path).values
        return data

    def load_random(self, num_tickers: int) -> np.ndarray:
        tickers = np.random.choice(self.files, num_tickers)
        return self.load_from_tickers(ticker_names=tickers)

    def load_all_from_dir(self):
        return self.load_from_tickers(ticker_names=self.files)

    def __file_exists(self, path) -> bool:
        return exists(path)

    def __load_file(self, path) -> pd.DataFrame:
        df = pd.read_csv(path)[["open", "high", "low", "close"]]
        if len(self.sma_periods) > 0:
            df = SMA(df, self.sma_periods, column="close")
        if self.bb_bands:
            df = std_bands(data=df)
        return df.dropna().reset_index(drop=True)

