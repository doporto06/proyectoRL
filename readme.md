## Instalación Gym y MuJoCo

#### 1. Seguir tutorial para instalar conda y gymnasium (windows): https://www.youtube.com/watch?v=gMgj4pSHLww

#### 2. En el mismo anaconda prompt, instalar stable-baselines3 (para usar los métodos de RL): 
```pip install stable-baselines3[extra]```

#### 3. instalar el repositorio del ambiente gym de cassie: https://github.com/perrin-isir/gym-cassie-run (yo segui las instrucciones del pip install dentro de mi ambiente conda)

#### 4. nose porque pero a mi no se me instaló la carpeta assets que debería estar en gym_cassie_run/env y me tiró un error la primera vez, basta con descargar el repo y pegar la carpeta en ese directorio (donde estan los archivos cassie_run.py y __init__.py)

#### 5. ejecutar el random-agent.py para ver si se instaló correctamente (si da error AttributeError: 'mujoco._structs.MjData' object has no attribute 'solver_iter'. Did you mean: 'solver_niter'?, ir al archivo mujoco_rendering.py y cambiar solver_iter por solver_niter)
## Repositorios de Cassie Robot

#### Repositorio oficial en Mujoco creado por "Agility Robotics": https://github.com/osudrl/cassie-mujoco-sim
#### Repositorio similar creado por "siekmanj", ingeniero en software de Agility Robotics: https://github.com/siekmanj/cassie
#### Repositorio similar creado por "Perrin-Gilbert": https://github.com/perrin-isir/gym-cassie-run
