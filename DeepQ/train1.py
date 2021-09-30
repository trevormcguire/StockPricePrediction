from environments import MultiTimeframe
from utils.load import DataLoader
from agents.Fatcat1 import Agent
import numpy as np

MODELDIR = "Results"
DATADIR = "PriceData"
num_tickers = int(input("Choose Number of Tickers: "))

loader = DataLoader(data_dir=DATADIR, bb_bands=False)
data = loader.load_random(num_tickers=num_tickers)

num_episodes = int(input("How many episodes? "))



env = MultiTimeframe(data=data, 
                     timeframes=[10,20,60], 
                     buy_on="open", 
                     sell_on="close")

agent = Agent(num_feats=4, 
              num_actions=3, 
              max_replay_mem=2000, 
              target_update_thresh=100, 
              replay_batch_size=32)


for n in range(num_episodes):
    env.reset()
    done = False #terminating conditions are if we exit our position or if we reach end of the timeseries
    while not done:
        current_state = env.get_state()
        action = agent.action(current_state)
        reward, next_state, done = env.step(action)
        if next_state != None:
            agent.replay_memory.append((current_state, action, reward, next_state, done))
            agent.replay() #will only replay if memory > replay_batch_size
            if done:
                agent.decay_epsilon() #episode finished, decay epsilon

    if n % 1000 == 0 and n > 0:
        agent.save_model(f"{MODELDIR}/fatcat1_e{n}_p{sum(env.episode_profits)*100:.2f}.pt")

print(f"Finished training after {num_episodes} trades.")
print(f"Total profits: {sum(env.episode_profits)*100:.2f}")
print(f"Mean Profit: {np.mean(env.episode_profits)*100:.2f}")

with open(f"{MODELDIR}/fatcat1_e{num_episodes}_summary.txt", "w") as f:
    f.write(str(env.episode_profits))
print("Saved profits over time in ", MODELDIR)
