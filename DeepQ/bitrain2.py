from environments import MultiTimeframePPT
from utils.load import DataLoader
from agents.Fatcat1 import Agent
import numpy as np
from copy import deepcopy

MODELDIR = "Results/bitestppt"
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
                     sell_on="open")

buyer = Agent(num_feats=11, 
              num_actions=2, 
              max_replay_mem=2000, 
              target_update_thresh=100, 
              replay_batch_size=32)

seller = Agent(num_feats=11, 
               num_actions=2, 
               max_replay_mem=2000, 
               target_update_thresh=100, 
               replay_batch_size=32)


#technically if the buyer buys, the seller can still action that timestep since our execution times are T+1
#should buyer be blind to anything after buying? -- as its no longer its problem?

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

        if not env.holding_position:
            action = buyer.action(current_state)
            reward, next_state, done = env.step(action)
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
        buyer.save_model(f"{MODELDIR}/bifatcat2_buyer_e{n}_p{sum(env.episode_profits)*100:.2f}.pt")
        seller.save_model(f"{MODELDIR}/bifatcat2_seller_e{n}_p{sum(env.episode_profits)*100:.2f}.pt")

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

with open(f"{MODELDIR}/bifatcat2_e{num_episodes}_summary.txt", "w") as f:
    f.write(str(data2write))

print("Saved datapoints @ ", MODELDIR)