import gymnasium as gym
import gym_cassie_run
env = gym.make("CassieRun-v0", render_mode='human')
observation, info = env.reset()
for _ in range(2000):
   print(_)
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
