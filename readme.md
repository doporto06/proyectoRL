## Instalación Gym y MuJoCo

#### 1. Seguir tutorial para instalar conda y gymnasium (windows): https://www.youtube.com/watch?v=gMgj4pSHLww

#### 2. En el mismo anaconda prompt, instalar stable-baselines3 (para usar los métodos de RL): 
```pip install stable-baselines3[extra]```

#### 3. ejecutar el random-agent.py para ver si se instaló correctamente (si da error AttributeError: 'mujoco._structs.MjData' object has no attribute 'solver_iter'. Did you mean: 'solver_niter'?, ir al archivo mujoco_rendering.py y cambiar solver_iter por solver_niter)