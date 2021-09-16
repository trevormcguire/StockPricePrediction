from os import listdir
from typing import *
import pandas as pd
import numpy as np
from functools import reduce
from random import sample

def pct_change(arr: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    return np.diff(arr, axis=0) / arr[:-1]

def choose_tickers(PATH: str, num_tickers: int) -> List:
    files = listdir(PATH)
    return [x.replace(".csv", "") for x in sample(files, num_tickers)]
    
class TickerDataGenerator:
    """
    Class to Generate Train and Test Data from a list of tickers and a directory.
    ---------
    Requirements: csv files of price data stored within a single directory (data_dir param)
    ---------
    PARAMS
        1. tickers: a list of tickers used to create training data from
        2. data_path: directory where data is stored in csv's
    ---------
    """
    def __init__(self, tickers: Union[list,str], data_dir: str):
        if type(tickers) is str:
            tickers = [tickers]
        tickers = [t.replace(".csv", "") for t in tickers]
        self.path = data_dir
        self.tickers = tickers
        self.__check_files()

    def __call__(self) -> pd.DataFrame:
        return self.generate_data()

    def generate_data(self) -> pd.DataFrame:
        #********** Helper functions **************
        def filter_by_date(df: pd.DataFrame) -> pd.DataFrame:
            return df[(df.date >= start) & (df.date <= end)].reset_index(drop=True)
                    #-----------------
        def rename_cols(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
            old_cols = [col for col in df.columns if col != "date"]
            new_cols = [f"{ticker}_{col}" for col in old_cols]
            return df.rename(columns=dict(zip(old_cols, new_cols)))
                    #-----------------
        def as_pct_change(df: pd.DataFrame) -> pd.DataFrame:
            df.close = np.cumsum(df.close.pct_change)
            df.dropna(inplace=True)
            return df
        #******************************************
        df_list = []
        for ticker in self.tickers:
            df_list.append(self.timestamp_to_date(self.__load_data(ticker)))
        start = max([df.date.min() for df in df_list])
        end = min([df.date.max() for df in df_list])

        df_list = [filter_by_date(df) for df in df_list]
        df_list = [rename_cols(ticker, df) for ticker, df in list(zip(self.tickers, df_list))]

        return reduce(lambda x, y: pd.merge(x, y, on = 'date'), df_list)

    def __load_data(self, ticker: str) -> pd.DataFrame:
        assert f"{ticker}.csv" in listdir(self.path), f"{ticker} not in {self.path}"
        return pd.read_csv(f"{self.path}/{ticker}.csv")

    def __check_files(self):
        files = listdir(self.path)
        for ticker in self.tickers:
            assert f"{ticker}.csv" in files, f"{ticker}.csv not in {self.path}"
                
    def timestamp_to_date(self, df: pd.DataFrame, as_index: bool = False) -> pd.DataFrame:
        df.date = pd.to_datetime(df.date).apply(lambda d: d.date())
        if as_index:
            df.set_index('date', inplace=True)
        return df

    def __train_test_split(self, arr: np.ndarray, period: int, split_perc: float) -> Tuple[np.ndarray]:
        if type(arr) is pd.DataFrame:
            arr = arr.values

        data = []
        for idx in range(len(arr) - period):
            data.append(arr[idx: idx+period])
            
        data = np.array(data)
        test_size = int(data.shape[0] * split_perc)
        train_size = data.shape[0] - test_size

        X_train, X_test = data[:train_size,:-1,:], data[train_size:,:-1]
        y_train, y_test = data[:train_size,-1,:], data[train_size:,-1,:]

        return X_train, y_train, X_test, y_test

    def create_train_data(self, 
                          feature_cols: List[Text], 
                          label_col: str,
                          split_perc: float,
                          period: int,
                          as_pct_change: bool, 
                          as_cumsum: bool) -> Tuple[Union[np.ndarray, List]]:
        #********** Helper functions **************
                    #-----------------
        def column_handler():
            features, allowed_cols = {}, ['close', 'open' ,'high', 'low', 'date', 'volume']
            for f in feature_cols:
                assert f in allowed_cols, f"{f} not found. Must be in {allowed_cols}"
                features[f] = [x for x in df.columns if x.find(f) != -1]
            return features
                    #-----------------
        def split_by_ticker(df: pd.DataFrame) -> list:
            ticker_data = []
            for ticker in self.tickers:
                ticker_data.append(df[[x for x in df.columns if x.find(ticker) != -1]])
            return ticker_data
                    #-----------------
        #******************************************
        assert label_col in feature_cols, f"{label_col} must be included in {feature_cols}"
        assert 0.0 <= split_perc <= 1.0, "Split percentage must be between 0 and 1"

        feature_cols = [x.lower() for x in feature_cols]
        df = self.generate_data()
        feature_cols = column_handler()
        df = df[np.array(list(feature_cols.values())).flatten()] #filter out columns we don't want
        df = split_by_ticker(df) #splits data by ticker so we can generate training data for each

        label_feature_index = [x.split("_")[1] for x in df[0]].index(label_col) #get col index of our y value

        if as_pct_change:
            df = [pct_change(d) for d in df] #handles for each column
        if as_cumsum:
            df = [np.cumsum(d, axis=0) for d in df] #handles for each column

        #xtrain/ytrain can be one big array but ytest/xtest should be a list for each ticker so we can visualize later
        all_X_train, all_y_train, all_X_test, all_y_test = [],[],[],[]
        for data in df:
            X_train, y_train, X_test, y_test = self.__train_test_split(data, period, split_perc)
            all_X_train.append(X_train)
            all_y_train.append(y_train[:,label_feature_index]) #filter for only label column
            all_X_test.append(X_test)
            all_y_test.append(y_test[:,label_feature_index])

        return np.concatenate(all_X_train), np.concatenate(all_y_train), all_X_test, all_y_test
        
