import os
import gymnasium as gym
import gym_cassie_run
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

if __name__ == '__main__':
   env = gym.make("CassieRun-v0", render_mode=None)
   log_path = os.path.join("Modelos", "SAC", "SAC-01")
   #model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_path) #, learning_rate=0.00003
   model = SAC.load("./Modelos/SAC/SAC-01/model_10730000_steps", env=env)
   eval_callback = EvalCallback(env, best_model_save_path=log_path,
                                 log_path=log_path, eval_freq=5000,
                                 deterministic=True, render=False)
   checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./Modelos/SAC/SAC-01',
                                       name_prefix='model')
   #model.learn(total_timesteps=5_000_000, callback=[eval_callback, checkpoint_callback])
   remaining_timesteps = 15_000_000 - 10_730_000
   model.learn(total_timesteps=remaining_timesteps, log_interval=1, callback=[eval_callback, checkpoint_callback], reset_num_timesteps=False) 
   model_path = os.path.join("Modelos", "SAC", "SAC-01")
   model.save(model_path)

   env.close()