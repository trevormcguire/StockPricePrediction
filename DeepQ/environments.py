import numpy as np 
from typing import *
import utils.functional as F 
from rewards import Reward, WeightedMean

class Environment(object):
    """
    ------
    PARAMS
    ------
        1. buy_on -> must be open or close
        2. sell_on -> must be open or close
        3. dictionary of ticker: prices
                'prices' must be numpy array
        4. rand_start -> if True (default), selects a random starting idx
        5. period -> amount of trading days for agent to consider

    ------
        Note: Ensure OHLC is the first 4 values of last dimension in array passed
    """
    def __init__(self, 
                 data: dict, 
                 buy_on: str, 
                 sell_on: str,
                 period: int,
                 reward_function: Callable = Reward(),
                 rand_start: bool = True):
        assert type(data) is dict, "data must be a dictionary of ticker: prices where prices is numpy array"
        assert buy_on.lower() in ["open", "close"], f"buy on must be eiter open or close. You passed {buy_on}"
        assert sell_on.lower() in ["open", "close"], f"buy on must be eiter open or close. You passed {sell_on}"
        self.rand_start = rand_start
        self.period = period
        self.data = data
        self.holding_position = False
        self.buy_on = buy_on.lower()
        self.sell_on = sell_on.lower()
        self.actions = {"hold": 0, "enter": 1, "exit": 2}
        self.reward_handler = reward_function
        self.reset()

    def reset(self):
        """
        Switch ticker we are working with randomly and reset environment 
        """
        self.ticker = np.random.choice(list(self.data.keys()))
        self.prices = self.data[self.ticker]
        self.len_prices = len(self.prices)
        self.holding_position = False
        self.entry_price, self.exit_price = None, None
        self.curr_idx = np.random.choice(np.arange(self.len_prices - (self.period * 2))) if self.rand_start else 0 

    def get_state(self):
        state = self.prices[self.curr_idx: self.curr_idx+self.period]
        self.curr_ohlc = state[-1][:4] #last row
        #have to make next ohlc independent of get_next_state() due to transformations like in multitimeframe
        self.next_ohlc = self.prices[self.curr_idx+1: self.curr_idx + 1 + self.period][0][:4] #first row
        return state

    def get_next_state(self):
        return self.prices[self.curr_idx + 1: self.curr_idx + 1 + self.period]

    # def is_done(self) -> Tuple[bool, np.ndarray]:
    #     """
    #     checks which index our next state will be at
    #     if at end of time series returns done=True, next_state=None
    #     if we aren't at the end, returns done=False, next_state
    #     """
    #     done = False
    #     if self.curr_idx + 1 > self.len_prices - self.period: #test if next state will out run the length of our data
    #         done = True
    #         next_state = None #reached end of timeseries, next is None
    #     else:
    #         next_state = self.get_next_state()
    #     return done, next_state

    def step(self):
        raise NotImplementedError
        


class MultiTimeframe(Environment):
    """
    ------
    PARAMS
    ------
        1. timeframes -> List of timeframes to consider (in trading days)
                Example: [10, 20, 60] -> will consider 10d, 20d, 60d
        2. buy_on -> must be open or close. If 'open', we will buy on the next open.
        3. sell_on -> must be open or close. If 'open', we will sell on next open.
        4. dictionary of ticker: prices, where 'prices' must be numpy array
        5. rand_start -> if True (default), selects a random starting idx
    """
    def __init__(self, 
                 data: dict,
                 timeframes: List[int], 
                 buy_on: str, 
                 sell_on: str, 
                 reward_function: Callable = WeightedMean(),
                 rand_start: bool = True):
        period = max(timeframes)
        super(MultiTimeframe, self).__init__(data, buy_on, sell_on, period, reward_function, rand_start)
        self.timeframes = timeframes
        self.timeframes.sort()
        self.action_map = {
            0: "hold", 
            1: "enter", 
            2: "exit"
            }
        self.ppt = [] #profit per tick
        self.reward_handler = reward_function
        self.episode_profits = []
    
    def get_state(self):
        state = super().get_state()
        return F.state_to_timeframes(state, self.timeframes)

    def get_next_state(self):
        next_state = super().get_next_state()
        return F.state_to_timeframes(next_state, self.timeframes)

    def __get_curr_price(self):
        if self.buy_on == "close":
            curr_price = self.curr_ohlc[-1]
        else:
            curr_price = self.next_ohlc[0]
        return curr_price
    
    def step(self, action: int) -> Tuple:
        """
        Terminating conditions:
            1. Exit Position
            2. Reach end of time series
        """
        reward = 0.0
        terminate = False
        action = self.action_map[action]

        if self.holding_position:
            curr_price = self.__get_curr_price()
            self.ppt.append(((curr_price - self.entry_price) / self.entry_price) * 100)

        if action == "enter" and not self.holding_position:
            self.entry_price = self.__get_curr_price()
            self.holding_position = True
            print(f"Bought {self.ticker} @ {self.entry_price} ||")

        elif action == "exit" and self.holding_position:
            self.exit_price = curr_price
            profit = (self.exit_price - self.entry_price) / self.entry_price
            self.episode_profits.append(profit)
            #---Calculate reward function here---
            reward = self.reward_handler.give_reward(self.ppt, profit)
            #------------------------------------
            terminate = True
            self.holding_position = False
            self.ppt = [] #reset
            print(f"Sold {self.ticker} @ {self.exit_price} || Profit: {profit * 100:.2f}")
        

        if self.curr_idx + 1 > self.len_prices - self.period: #test if next state will out run the length of our data
            terminate = True
            next_state = None #reached end of timeseries, next is None
        else:
            next_state = self.get_next_state()

        if not terminate:
            self.curr_idx += 1
        
        return reward, next_state, terminate




class MultiTimeframePPT(MultiTimeframe):
    """
    ------
    PARAMS
    ------
        1. timeframes -> List of timeframes to consider (in trading days)
                Example: [10, 20, 60] -> will consider 10d, 20d, 60d
        2. buy_on -> must be open or close. If 'open', we will buy on the next open.
        3. sell_on -> must be open or close. If 'open', we will sell on next open.
        4. dictionary of ticker: prices, where 'prices' must be numpy array
        5. rand_start -> if True (default), selects a random starting idx
    """
    def __init__(self, 
                 data: dict,
                 timeframes: List[int], 
                 buy_on: str, 
                 sell_on: str, 
                 reward_function: Callable = Reward(),
                 rand_start: bool = True):
        super(MultiTimeframePPT, self).__init__(data, timeframes, buy_on, sell_on, reward_function, rand_start)
    
    def __get_curr_price(self):
        if self.buy_on == "close":
            curr_price = self.curr_ohlc[-1]
        else:
            curr_price = self.next_ohlc[0]
        return curr_price

    def step(self, action: int) -> Tuple:
        """
        Terminating conditions:
            1. Exit Position
            2. Reach end of time series
        """
        reward = 0.0
        terminate = False
        action = self.action_map[action]

        if self.holding_position:
            curr_price = self.__get_curr_price()
            self.ppt.append(((curr_price - self.entry_price) / self.entry_price) * 100)
            #reward = F.tanh(self.ppt[-1])
            reward = F.tanh(self.reward_handler.give_reward(self.ppt)) #iwm for holds too

        if action == "enter" and not self.holding_position:
            self.entry_price = self.__get_curr_price()
            self.holding_position = True
            print(f"Bought {self.ticker} @ {self.entry_price} ||")

        elif action == "exit" and self.holding_position:
            self.exit_price = curr_price
            profit = (self.exit_price - self.entry_price) / self.entry_price
            self.episode_profits.append(profit)
            #---Calculate reward function here---
            reward = F.tanh(self.reward_handler.give_reward(self.ppt, profit * 100))
            #------------------------------------
            terminate = True
            self.holding_position = False
            self.ppt = [] #reset
            print(f"Sold {self.ticker} @ {self.exit_price} || Profit: {profit * 100:.2f}")
        
        if self.curr_idx + 1 > self.len_prices - self.period: #test if next state will out run the length of our data
            terminate = True
            next_state = None #reached end of timeseries, next is None
        else:
            next_state = self.get_next_state()

        if not terminate:
            self.curr_idx += 1
        
        return reward, next_state, terminate

