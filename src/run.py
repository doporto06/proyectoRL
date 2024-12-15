import gymnasium as gym
import gym_cassie_run
from stable_baselines3 import PPO, SAC

#model = PPO.load("Modelos/PPO/PPO-01/best_model") 
model = SAC.load("Modelos/SAC/SAC-01/best_model")

env = gym.make("CassieRun-v0", render_mode='human')
obs, info = env.reset()
terminated = False
truncated = False
while True:
   action, state = model.predict(obs)
   obs, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      obs, info = env.reset()

env.close()