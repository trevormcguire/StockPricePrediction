import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import *

from os import listdir
import random
import sys
#****************************************************************************
#****************************************************************************

#global vars
PATH = "PriceData"
MODEL_DIR = "FatCat"
WINDOW_SIZE = 30
TICKER = input("Which ticker do you want to run for?: ").upper()
assert f"{TICKER}.csv" in listdir(PATH), f"Could not find {TICKER} in {PATH}"
EPOCHS = 10
BATCH_SIZE = 32

#****************************************************************************

def load_ticker_data(ticker: str, 
                     cols: list = None, 
                     as_pct: bool = True) -> Union[pd.DataFrame, np.ndarray]:
    ticker = ticker.replace(".csv", "")
    df = pd.read_csv(f"{PATH}/{ticker}.csv").round(2)
    if cols:
        for c in cols:
            assert c in df.columns, f"{c} not in dataframe columns"
        df = df[cols]
    if as_pct:
        df = np.cumsum(df.pct_change().dropna().reset_index(drop=True))
    return df.values[20:] #knock off the stock's start

def sigmoid(x):
    return 1/(1+np.exp(-x))

#****************************************************************************

class FatCatBrain(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(FatCatBrain, self).__init__()
        self.num_gru_layers = 2
        self.GRU = nn.GRU(input_dim, 32, self.num_gru_layers, batch_first=True)
        
        self.relu = nn.ReLU()

        self.dense1 = nn.Linear(32, 8)
        self.out_dense = nn.Linear(8, output_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        gru_hidden = torch.zeros(self.num_gru_layers, X.size(0), 32)
        out, _ = self.GRU(X, gru_hidden)
        out = out[:, -1, :]
        out = self.relu(self.dense1(out))
        out = self.out_dense(out)
        return out

#****************************************************************************

class FatCat(object):
    def __init__(self,
                 context_period: int,
                 model_dir: str,
                 num_actions: int = 3, 
                 inference_mode: bool = False):
        
        self.model_dir = model_dir
        self.context_period = context_period
        self.num_actions = num_actions

        self.model = FatCatBrain(4, num_actions)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)


        self.gamma = 0.9 #how much importance we give future rewards #r1 + gamma*r2 + gamma^2*r3 + gamma^3*r4 ...
        self.epsilon = 0.9 #marks the boundary between expoitation and exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.96
        self.inventory, self.memory = [], []

        self.inference_mode = inference_mode

    def __ensure_is_tensor(self, matrix: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return torch.Tensor(matrix) if type(matrix) is np.ndarray else matrix
    
    def train_model(self, 
                    num_epochs: int, 
                    X: Union[np.ndarray, torch.Tensor],
                    y: Union[np.ndarray, torch.Tensor]):
        
        X, y = self.__ensure_is_tensor(X), self.__ensure_is_tensor(y)

        for e in range(num_epochs):
            self.optimizer.zero_grad()
            yhat = self.model(X)
            loss = self.criterion(yhat, y)
            loss.backward()
            self.optimizer.step()
        

    def predict(self, 
                X: Union[np.ndarray, torch.Tensor], 
                logits: bool = False) -> Union[int, torch.Tensor]:
        X = self.__ensure_is_tensor(X)
        with torch.no_grad():
            if logits:
                return self.model(X)
            return torch.argmax(self.model(X)).item()


    def action(self, state: Union[np.ndarray, torch.Tensor]):
        if not self.inference_mode and random.random() <= self.epsilon:
            return random.randrange(self.num_actions) #rand int between 0 - 2
        state = self.__ensure_is_tensor(state)
        return self.predict(state)

    def study(self, batch_size: int):
        batch, mem_len = [], len(self.memory)
        for idx in range(mem_len - batch_size + 1, mem_len):
            batch.append(self.memory[idx])

        for _, curr_state, action, reward, next_state in batch:
            curr_state, next_state = self.__ensure_is_tensor(curr_state), self.__ensure_is_tensor(next_state)
            #if agent is wrong, reward will be 0. This is where the RL happens, as positive reward is added for torch.argmax eventually
            target = reward + self.gamma * torch.max(self.predict(next_state, logits=True)).item()
            curr_pred = self.predict(curr_state, logits=True)
            curr_pred[0][action] = target #update the current predicted action with the target from next_state
            self.train_model(1, curr_state, curr_pred)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, name: str):
        name = name.replace(".pt", "")
        torch.save(self.model.state_dict(), f"{self.model_dir}/{name}.pt")
        print(f"Saved model at {self.model_dir}/{name}.pt")

    def load_model(self, name: str):
        name = name.replace(".pt", "")
        self.model.load_state_dict(torch.load(f"{self.model_dir}/{name}.pt"))

    def save_memory(self, path: str):
        if path.find(".csv") == -1:
            path = path + ".csv"
        df = pd.DataFrame(self.memory, columns=["curr_price", "curr_state", "action", "reward", "next_state"])
        df.to_csv(f"{self.model_dir}/{path}")
        print(f"Saved memory at {self.model_dir}/{path}")


#****************************************************************************
#****************************************************************************

data = load_ticker_data(TICKER, cols=["open","high", "low","close"])

epochs = 10
agent = FatCat(WINDOW_SIZE, MODEL_DIR)

for e in range(epochs + 1):
    total_profit = 0
    agent.inventory = [] #reset inventory
    agent.memory = [] #reset memory, as it was saved as a csv in MODEL_DIR
    for idx in range(len(data) - WINDOW_SIZE - 2):
        reward, curr_price = 0, data[idx][-1]
        state = sigmoid(data[idx:idx+WINDOW_SIZE])
        state = np.array([state])

        action = agent.action(state)
        
        if action == 1: #buy
            agent.inventory.append(curr_price)
            print(f"Bought @ {curr_price}")
        elif action == 2 and len(agent.inventory) > 0: #sell
            bought_price = agent.inventory.pop(0)
            profit = curr_price - bought_price
            #reward = max(profit, 0)
            reward = profit #penalize bad decisions, reward good decisions
            total_profit += profit
            print(f"Sold @ {curr_price} || Profit: {profit * 100:.2f}%")

        next_state = sigmoid(data[idx+1:idx+1+WINDOW_SIZE]) #we need next state to reinforce the reward we just got
        next_state = np.array([next_state])
        agent.memory.append((curr_price, state, action, reward, next_state))
        if len(agent.memory) > BATCH_SIZE:
            agent.study(BATCH_SIZE)
    print("*"*30)
    print(f"Total Profit: {total_profit * 100:.2f}%")
    print("*"*30)
    agent.save_model(f"FatCat_DRL_{TICKER}_e{e}")
    agent.save_memory(f"FatCat_DRL_{TICKER}_mem_e{e}.csv")

