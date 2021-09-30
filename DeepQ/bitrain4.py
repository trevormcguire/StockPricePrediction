from environments import MultiTimeframePPT
from utils.load import DataLoader
from agents.Fatcat1 import Agent
import numpy as np
from copy import deepcopy
import rewards as R
import utils.functional as F 

lobf = F.ExtremaRegression(start_period = 7, min_period = 2)

def stack_regression(state):
    if state == None:
        return state
    arr2append = []
    for splice in state:
        batch_comps = []
        for comp in splice:
            close = comp[:,3]
            highs = comp[:,1]
            lows = comp[:,2]
            resistanceh = lobf(highs, kind="h").reshape(-1,1)
            supportl = lobf(lows, kind = "l").reshape(-1,1)
            resistancec = lobf(close, kind="h").reshape(-1,1)
            supportc = lobf(close, kind = "l").reshape(-1,1)
            batch_comps.append(np.hstack([supportl, resistanceh, supportc, resistancec]))
        batch_comps = np.array(batch_comps)
        arr2append.append(batch_comps)
    return [ np.dstack((x,y)) for x,y in zip(state, arr2append) ]


MODELDIR = "Results/bitest5"
DATADIR = "PriceData"
num_tickers = int(input("Choose Number of Tickers: "))


loader = DataLoader(data_dir=DATADIR, 
                    sma_periods=[9, 20, 50, 100, 200], 
                    bb_bands=True)

data = loader.load_random(num_tickers=num_tickers)

num_episodes = int(input("How many episodes? "))

env = MultiTimeframePPT(data=data, 
                        timeframes=[10,20,60], 
                        buy_on="open", 
                        sell_on="open", 
                        reward_function=R.AdjustedWeightedMean())

buyer = Agent(num_feats=15, 
              num_actions=2, 
              max_replay_mem=4000, 
              target_update_thresh=150, 
              replay_batch_size=32, 
              gamma=0.15)

seller = Agent(num_feats=15, 
               num_actions=2, 
               max_replay_mem=4000, 
               target_update_thresh=150, 
               replay_batch_size=32, 
               gamma=0.15)

times_held = []
for n in range(num_episodes):
    env.reset()
    done = False
    buyer_memory = []
    seller_memory = []
    bought_state = None
    bought_next_state = None
    time_held = 0
    while not done:
        current_state = env.get_state()
        current_state = stack_regression(current_state)
        

        if not env.holding_position:
            action = buyer.action(current_state)
            reward, next_state, done = env.step(action)
            next_state = stack_regression(next_state)

            if action == 1:
                #delay reward until we sell
                bought_state = deepcopy(current_state)
                bought_next_state = deepcopy(next_state)
            else:
                buyer_memory.append((current_state, action, reward, next_state, done))
            
        else:
            action = seller.action(current_state)
            if action == 1:
                action += 1
            reward, next_state, done = env.step(action)
            next_state = stack_regression(next_state)
            if action == 2:
                buyer_memory.append((bought_state, 1, reward, bought_next_state, done))
            
            seller_memory.append((current_state, action, reward, next_state, done))
        time_held += 1

    if next_state != None:
        buyer.replay_memory += buyer_memory
        seller.replay_memory += seller_memory

        buyer.replay()
        seller.replay()

        buyer.decay_epsilon()
        seller.decay_epsilon()
        times_held.append(time_held)

    if n % 1000 == 0 and n > 0:
        buyer.save_model(f"{MODELDIR}/bifatcat5_buyer_e{n}_p{sum(env.episode_profits)*100:.2f}.pt")
        seller.save_model(f"{MODELDIR}/bifatcat5_seller_e{n}_p{sum(env.episode_profits)*100:.2f}.pt")

print(f"Finished training after {num_episodes} trades.")
print(f"Total profits: {sum(env.episode_profits)*100:.2f}")
print(f"Mean Profit: {np.mean(env.episode_profits)*100:.2f}")
print(f"Average Time Held: {np.mean(times_held)}")

data2write = {}
data2write["Num Episodes"] = num_episodes
data2write["Total profits"] = sum(env.episode_profits)*100
data2write["Mean Profit"] = np.mean(env.episode_profits)*100
data2write["Average Time Held"] = np.mean(times_held)
data2write["Profits"] = env.episode_profits

with open(f"{MODELDIR}/bifatcat5_e{num_episodes}_summary.txt", "w") as f:
    f.write(str(data2write))

print("Saved datapoints @ ", MODELDIR)
