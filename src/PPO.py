import os
import gymnasium as gym
import gym_cassie_run
import numpy as np
import optuna
import glob
from IPython.display import HTML, Video
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env


'''
OPTIMIZACIÓN DE HIPERPARÁMETROS USANDO OPTUNA

# Definir todas las combinaciones válidas de n_steps y batch_size
valid_combinations = [
    {"n_steps": 128, "batch_size": 64}, {"n_steps": 256, "batch_size": 64},
    {"n_steps": 256, "batch_size": 128}, {"n_steps": 512, "batch_size": 64},
    {"n_steps": 512, "batch_size": 128}, {"n_steps": 512, "batch_size": 256},
    {"n_steps": 1024, "batch_size": 64}, {"n_steps": 1024, "batch_size": 128},
    {"n_steps": 1024, "batch_size": 256}, {"n_steps": 1024, "batch_size": 512},
    {"n_steps": 2048, "batch_size": 64}, {"n_steps": 2048, "batch_size": 128},
    {"n_steps": 2048, "batch_size": 256}, {"n_steps": 2048, "batch_size": 512}
]

n_trials = 1000   #Define cuantos intentos optimizará

def objective(trial):
    # Seleccionar una combinación válida
    combination = trial.suggest_categorical('n_steps_batch_size', valid_combinations)
    n_steps = combination["n_steps"]
    batch_size = combination["batch_size"]

    # Otros hiperparámetros
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.9999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.95, 0.99)

    print(f"Trial {trial.number + 1}/{n_trials}:")
    print(f"Hyperparameters: learning_rate={learning_rate}, gamma={gamma}, "
          f"n_steps={n_steps}, batch_size={batch_size}, gae_lambda={gae_lambda}")

    # Crear el entorno y el modelo
    env = Monitor(gym.make("CassieRun-v0", render_mode=None))
    try:
        model = PPO(
            policy='MlpPolicy',
            env=env,
            verbose=0,
            learning_rate=learning_rate,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            gae_lambda=gae_lambda,
        )
        model.learn(total_timesteps=10000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)
    finally:
        env.close()

    return -mean_reward

# Crear el estudio de Optuna y ejecutar la optimización
study = optuna.create_study(direction='minimize')
print("Starting optimization with Optuna...")
study.optimize(objective, n_trials=n_trials)

# Mostrar los mejores hiperparámetros encontrados
print("Mejores hiperparámetros:")
print(study.best_params)
'''


#Lista de Hiperparámetros "best_model_2"
#learning_rate=0.0001, n_steps=512, batch_size=256, gamma=0.975, gae_lambda=0.9686

#Lista de Hiperparámetros "best_model_3"
#learning_rate=0.000101, n_steps=1024, batch_size=512, gamma=0.975, gae_lambda=0.9686


#Una vez encontrados los hiperparámetros, se entrena el modelo
if __name__ == '__main__':
   log_path = os.path.join("Modelos entrenados")
   env = gym.make("CassieRun-v0", render_mode=None)

   #print("Tipo de espacio de acción:", type(env.action_space))
   #print("Espacio de acción:", env.action_space)
   #print("Límites de acción:", env.action_space.low)
   #print("Límites de acción:", env.action_space.high)
   
   #print("Tipo de espacio de observación:", type(env.observation_space))
   #print("Espacio de observación:", env.observation_space)
   #print("Límites de observación:", env.observation_space.low)
   #print("Límites de observación:", env.observation_space.high)


   log_path = os.path.join("Modelos entrenados")      # Carpeta para guardar los logs
   
   """ 
   PPO: Proximal Policy Optimization
   1. Política: Multilayer Perceptron (MlpPolicy)
   2. Entorno: env
   3. Imprime Información: verbose=0 (nada), verbose=1 (Básico), verbose=2 (Detallado)
   4. Dispositivo: device='cuda' (GPU), device='cpu' (CPU), device='auto' (Automático)
   5. Crea logs para TensorBoard en la carpeta especificada: tensorboard_log = log_path
   """
   
   #Hipermarámetros personalizados
   model = PPO(
    policy='MlpPolicy',
    env=env,
    learning_rate=0.000101,          # Tasa de aprendizaje
    n_steps=1024,                  # Número de pasos por actualización
    batch_size=512,                 # Tamaño del minibatch
    n_epochs=10,                   # Número de épocas para optimizar
    gamma=0.975,                    # Factor de descuento
    gae_lambda=0.9686,               # Parámetro GAE para trade-off entre sesgo y varianza
    clip_range=0.15,                # Parámetro de clipping para PPO
    clip_range_vf=None,            # Clipping para la función de valor (None = no clipping)
    normalize_advantage=True,      # Normalizar el advantage
    ent_coef=0.0,                  # Coeficiente de entropía para fomentar exploración
    vf_coef=0.5,                   # Coeficiente para la función de valor en la pérdida
    max_grad_norm=0.5,             # Clipping máximo de gradientes
    use_sde=False,                 # Exploración dependiente del estado (SDE)
    sde_sample_freq=-1,            # Frecuencia de muestreo para SDE (-1 = solo al inicio)
    target_kl=None,                # Límite para la divergencia KL
    stats_window_size=100,         # Tamaño de ventana para el logging de estadísticas
    tensorboard_log='logs/ppo',    # Ruta para guardar logs de TensorBoard
    policy_kwargs=None,            # Parámetros adicionales para la política
    verbose=1,                     # Nivel de verbosidad
    seed=42,                       # Semilla para reproducibilidad
    device='auto',                 # Usar GPU si está disponible
    _init_setup_model=True         # Inicializar el modelo al crear la instancia
   )
   
   # Callbacks
   eval_callback = EvalCallback(env, best_model_save_path=log_path,
                                 log_path=log_path, eval_freq=5000,
                                 deterministic=True, render=False)
   checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./Modelos entrenados',
                                       name_prefix='model')

   """ 
   Método Learn: Motor principal del entrenamiento en Stable Baselines 3
   1. Pasos del entorno. Número de episodios depende de cuántos 
   timesteps se necesitan para completar un episodio: total_timesteps=5_000_000
   2. Permite pasar funciones o listas de callbacks para 
   ejecutar durante el entrenamiento: callback=[eval_callback, checkpoint_callback] (2 callbacks)
   """

   model.learn(total_timesteps=19_000_000, callback=[eval_callback, checkpoint_callback])

   """ remaining_timesteps = 5_000_000 - 640_000
   model.learn(total_timesteps=remaining_timesteps, log_interval=1, callback=[eval_callback, checkpoint_callback], reset_num_timesteps=False) """
   model_path = os.path.join("Modelos entrenados")          #Carpeta para guardar el modelo entrenado
   model.save(model_path)                                   #Guardar el modelo entrenado
   env.close()
