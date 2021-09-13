from typing import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import sample
from os import listdir
import math
from copy import deepcopy

def choose_tickers(PATH: str, num_tickers: int) -> List[str]:
    files = listdir(PATH)
    return [x.replace(".csv", "") for x in sample(files, num_tickers)]


def std_bands(df: pd.DataFrame, periods: List[int] = [20], sigma: int = 2) -> pd.DataFrame:
    assert type(periods) in [list, tuple, set], "Periods must be iterable."
    for p in periods:
        df[f'LowerBand_{p}'] = df.SMA20 - (df.std20 * sigma)
        df[f'UpperBand_{p}'] = df.SMA20 + (df.std20 * sigma)
    return df

def get_candlestick_color(close: float, open: float) -> str:
    if close > open:
        return 'white'
    elif close < open:
        return 'black'
    else:
        return 'gray'

def candlestick_plot(data: pd.DataFrame, save_as: str = None):
    """
    Generate Candlestick Images
    ----
    PARAMS
    ----
        1. 'data' -> a full, or partial, of a pandas DataFrame (use .iloc[start:end] to get range)
        2. 'save_as' -> Full Path of where to save image (including name)
    -----
    """
    x = np.arange(len(data))
    fig, ax = plt.subplots(1, figsize=(3,3))
    for idx, val in data.iterrows():
        o,h,l,c = val['open'], val['high'], val['low'], val['close']
        clr = get_candlestick_color(c, o)
        x_idx = x[idx]
        plt.plot([x_idx, x_idx], [l, h], color=clr) #wick
        plt.plot([x_idx, x_idx], [o, o], color=clr) #open marker
        plt.plot([x_idx, x_idx], [c, c], color=clr) #close marker
        rect = mpl.patches.Rectangle((x_idx-0.5, o), 1, (c - o), facecolor=clr, edgecolor='black', linewidth=1, zorder=3)
        ax.add_patch(rect)
    plt.axis('off')
    if type(save_as) is str:
        plt.savefig(save_as, bbox_inches="tight", pad_inches = 0)
        plt.close()
    else:
        plt.show()



class GramianAngularField:
    """
    Gramian Angular Field (GAF)
    For images derived from time series, GAF represents a temporal correlation between time values
    -----
    PARAMS
    -----
        1. 'series' -> a 1D numpy array of time series data
    """
    def __init__(self, series: np.ndarray):
        self.series = series

    def __call__(self) -> np.ndarray:
        self.transform()
        return self.gaf

    def __tabulate(self, x, func) -> np.ndarray:
        return np.vectorize(func)(*np.meshgrid(x, x, sparse=True))

    def __cossum(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return math.cos(a+b)

    def __scaler(self, series: np.ndarray) -> np.ndarray:
        """
        MinMax Scaler
        """
        smallest, largest = np.min(series), np.max(series)
        return ((2*series - largest - smallest) / (largest - smallest)).astype(np.float16)

    def transform(self):
        polar_encoding = np.arccos(self.__scaler(self.series))
        self.gaf = self.__tabulate(x=polar_encoding, func=self.__cossum)

    def plot(self, color="g"):
        assert color in ["g", "h", "grey", "gray", "hot"], "color must be g (greyscale) or h (hot)"
        if color in ["g", "grey", "gray"]:
            clr = "Greys_r"
        else:
            clr = "hot"
        plt.imshow(self.gaf, cmap=clr, interpolation='nearest') #or hot
        plt.axis("off")
        plt.show()


class Conv2DHelper(object):
    """
    -----
    Functions to calculate output of Convolutional layers and Max Pooling layers in PyTorch
    Reference: 
        1. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        2. https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

    -----
    - Both MaxPooling2D and Conv2D have the same root calculation, which is as follows:
    
        -> (Dim[idx] + (2 * Padding[idx]) - (Dilation[idx] * KernelSize[idx] - 1)) / Stride + 1

    - To handle non-square images, the internal function __param_handler converts the necessary params to tuples.
    -----
    PARAMS
    -----
        1. 'dims' -> input dimension (height x width) -> represented in our functions as h and w
        2. 'kernel_size' -> kernel size of convolutional layer, can be int or tuple
        3. 'stride' -> stride size of convolutional layer, can be int or tuple. Default is 1.
        3. 'padding' -> default is 0
        5. 'dilation' -> default is 1
    ----
    #nn.Conv2d(1, 6, 5),

    output = 6 * conv_output[0] * conv_output[1]
    """
    def __init__(self, image_dims: Union[Tuple[int], int]):
        self.height, self.width = self.__param_handler(image_dims)
        self.OG_height = deepcopy(self.height)
        self.OG_width = deepcopy(self.width)
        self.input_channels = None

    def __param_handler(self, param: Union[Tuple[int], int]):
        if not type(param) is tuple:
            return (param, param)
        return param

    def __calculate_output(self,
                           kernel_size: Union[Tuple[int], int], 
                           stride: Union[Tuple[int], int] = 1,
                           padding: Union[Tuple[int], int] = 0, 
                           dilation: Union[Tuple[int], int] = 1):
        
        kernel_size, stride = self.__param_handler(kernel_size), self.__param_handler(stride)
        padding, dilation = self.__param_handler(padding), self.__param_handler(dilation)

        self.height = (self.height + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) // stride[0] + 1
        self.width = (self.width + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    def conv(self, 
             in_channels: int, 
             out_channels: int, 
             kernel_size: Union[Tuple[int], int], 
             stride: Union[Tuple[int], int] = 1,
             padding: Union[Tuple[int], int] = 0, 
             dilation: Union[Tuple[int], int] = 1) -> Tuple[int]:
        """
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1)
        """
        
        self.__calculate_output(kernel_size=kernel_size, 
                                stride=stride, 
                                padding=padding, 
                                dilation=dilation)
        
        self.input_channels = out_channels
        return out_channels, self.height, self.width

    def maxpool(self,
                kernel_size: Union[Tuple[int], int], 
                stride: Union[Tuple[int], int] = None, 
                padding: Union[Tuple[int], int] = 0, 
                dilation: Union[Tuple[int], int] = 1) -> Tuple[int]:
        """
        nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1)
        ----
        NOTABLE PARAMS
        ----
            1. 'in_channels' == out_channels of conv function
            2. 'strides' -> if None, defaults as == to kernel_size
        ----
        """
        if not stride:
            stride = deepcopy(kernel_size)

        self.__calculate_output(kernel_size=kernel_size, 
                                stride=stride, 
                                padding=padding, 
                                dilation=dilation)
        
        return self.input_channels, self.height, self.width

    def linear(self):
        return self.input_channels * self.height * self.width

