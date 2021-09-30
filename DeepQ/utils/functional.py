import numpy as np
from typing import *
from scipy.signal import argrelextrema
from scipy.stats import linregress

def sigmoid(x: float) -> float:
    return 1/(1+np.exp(-x))

def tanh(x: float) -> float:
    return np.tanh(x)


def normalize3D(arr: np.ndarray) -> np.ndarray:
    batches = []
    for x in arr:
        batches.append(normalize(x))
    return np.vstack(batches)

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    min-max normalization
    """
    # if len(arr.shape) == 3:
    #     return normalize3D(arr)

    amin, amax = np.min(arr), np.max(arr)
    return (arr - amin) / (amax - amin)


def inv_weighted_mean(arr: np.ndarray) -> float:
    """
    inverses the array to weigh earlier samples heavier than more recent ones
    grace_period? weights[:grace_period] = x 
            --> where x == weights[-1] | wieghts[0]?
    """
    arr_len = len(arr)
    weights = np.arange(arr_len)+1
    return np.sum((arr[::-1] * weights) / ((arr_len*(arr_len+1)) / 2))

def weighted_mean(arr: np.ndarray) -> float:
    """
    gets the weighted mean of a 1d array
    """
    arr_len = len(arr)
    weights = np.arange(arr_len)+1
    return np.sum((arr * weights) / ((arr_len*(arr_len+1)) / 2))


def state_to_timeframes(X: np.ndarray, timeframes: List[int]) -> Tuple[np.ndarray]:
    """
    -----
    PARAMS
    -----
        1. X -> numpy array to slice into smaller bits
        2. timeframes -> periods to slice
    -----
    Returns:
        -> a tuple of arrays with length == len(timeframes)
    -----
    """
    timeframes.sort()
    assert timeframes[-1] <= len(X), f"X (length={len(X)}) must be >= in size to largest timeframe passed ({timeframes[-1]})"
    states = []
    for t in timeframes:
        states.append(np.array([normalize(X[-t:])])) #take most (t) most recent idxs for each timeframe
    return tuple(states)


def fibonacci(end: int) -> List[int]:
    """
    Generates fibonacci sequence until param (end) is reached.
    """
    fibs = []
    a, b = 0, 1
    while (a < end):
        c = b  
        b = a  
        a = c + b  
        fibs.append(a)
    return fibs

def calc_epsilon_decay(epsilon: float, min_epsilon: float, decay_rate: float) -> int:
    """
    counts the number of iterations for epsilon to decay
    """
    ct = 0
    while epsilon > min_epsilon:
        epsilon *= decay_rate
        ct += 1
    return ct


class ExtremaRegression:
    """
    Performs Linear Regression to get line of best fit given highs or lows for timeseries data
    -------
    PARAMS
    -------
        1. start_period -> should be <= len(data) - 1
        2. min_periods -> fallback periods if algorithm doesn't find line of best fit
    -------
    Example Usage:
        where 'data' is a 2D array of OHLC data...

        highs, lows = data[:,1], data[:,2]

        lobf = ExtremaRegression(start_period = 9, min_period = 2)
        resistance = lobf(highs, kind="h")
        support = lobf(lows, kind = "l")

        plt.plot(highs)
        plt.plot(lows)

        plt.plot(resistance, c="r")
        plt.plot(support, c="g")
        plt.show()
    """
    def __init__(self, 
                 start_period: int = 9, 
                 min_period: int = 2):
        
        self.start_period = start_period
        self.min_period = min_period

    def __call__(self, arr: np.ndarray, kind: str) -> np.ndarray:
        return self.transform(arr, kind)

    def transform(self, arr: np.ndarray, kind: str) -> np.ndarray:
        """
        ---------
        PARAMS
        ---------
            1. 'arr' -> an array of highs or lows for any ticker
            2. 'kind' -> Whether to consider maxima or minima
        ---------
        """
        kind = kind.lower()
        assert kind in ["highs", "high", "h", "lows", "low", "l"], f"param 'kind' must be either highs or lows"
        extrema = self.__get_local_extrema(arr=arr, 
                                           order=self.start_period, 
                                           fallback=self.min_period, 
                                           kind=kind)
        
        if len(extrema) == 0:
            return np.zeros(len(arr))
        start = np.min(extrema) #idx of first local extrema
 
        m = linregress(extrema, arr[extrema]).slope

        lobf = np.zeros(len(arr)) #line of best fit
        lobf[start] = arr[start] #start price at start idx
        lobf[start+1:] = m
        return np.cumsum(lobf)


    def __get_local_extrema(self, 
                            arr: np.ndarray,
                            order: int,
                            fallback: int,
                            kind: str) -> np.ndarray:
        #----------------------------------
        def calc_extrema(order_pd):
            if kind in ["highs", "high", "h"]:
                return argrelextrema(arr, np.greater, order=order_pd)[0]
            return argrelextrema(arr, np.less, order=order_pd)[0]
        #----------------------------------
        arr_len = 0
        while True:
            extrema = calc_extrema(order)
            arr_len = len(extrema)
            if 2 <= arr_len <= 4:
                break
            order -= 1
            if order == 0:
                extrema = calc_extrema(fallback)
                break
        return extrema


