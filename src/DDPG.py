import os
import optuna
import gymnasium as gym
import gym_cassie_run
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Definición de la función objetivo para Optuna
def objective(trial):
    # Definir los hiperparámetros a optimizar
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    #n_steps = trial.suggest_categorical('n_steps', [128, 256, 512, 1024, 2048])

    # Crear el entorno
    env = gym.make("CassieRun-v0", render_mode=None)

    # Crear el modelo con los hiperparámetros sugeridos
    model = DDPG('MlpPolicy', env, verbose=0, learning_rate=learning_rate, gamma=gamma)

    # Entrenar el modelo por un número limitado de timesteps
    model.learn(total_timesteps=50000)

    # Evaluar el modelo
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    # Optuna minimiza por defecto, así que devolvemos -mean_reward para maximizar la recompensa
    return -mean_reward

if __name__ == '__main__':
   # Crear el estudio de Optuna y ejecutar la optimización
   #study = optuna.create_study(direction='minimize')
   #study.optimize(objective, n_trials=20)

   # Mostrar los mejores hiperparámetros encontrados
   #print("Mejores hiperparámetros:")
   #print(study.best_params)

   
   env = gym.make("CassieRun-v0", render_mode=None)
   log_path = os.path.join("Modelos", "DDPG", "DDPG-09")
   #model = DDPG('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_path, learning_rate = 0.0003)
   model = DDPG.load("./Modelos/DDPG/DDPG-09/model_2330000_steps", env=env)
   eval_callback = EvalCallback(env, best_model_save_path=log_path,
                                 log_path=log_path, eval_freq=5000,
                                 deterministic=True, render=False)
   checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./Modelos/DDPG/DDPG-09',
                                       name_prefix='model')
   #model.learn(total_timesteps=20_000_000, callback=[eval_callback, checkpoint_callback])
   remaining_timesteps = 20_000_000 - 2_330_000
   model.learn(total_timesteps=remaining_timesteps, log_interval=1, callback=[eval_callback, checkpoint_callback], reset_num_timesteps=False)
   model_path = os.path.join("Modelos", "DDPG", "DDPG-09")
   model.save(model_path)

   env.close()
   
