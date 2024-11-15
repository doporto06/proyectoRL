## Uso de códigos
PPO.py, SAC.py y DDPG.py cada uno corresponde al archivo para entrenar al robot con el método correspondiente. Para entrenar al robot con uno de los métodos basta con ejecutar el archivo con: 

```bash
python [metodo].py
```

Al momento de ejecutar el archivo, se creará un directorio llamado "Modelos/[metodo]/[metodo]-01" (Ej: Modelos/PPO/PPO-01).

Mientras se esté ejecutando, dentro de [metodo]-01 se irán guardando los modelos en los distintos steps de ejecución (piensen que son como los episodios), junto con un archivo llamado best_model.zip que irá guardando el mejor modelo hasta el momento. Para ver el desempeño del robot en el entrenamiento tienen que abrir otra terminal en la carpeta [metodo]-01 y escribir (por ejemplo en el caso de PPO, es análogo a los demás):

```bash
tensorboard --logdir=PPO_1
```
Esto ejecutará un dashboard de Tensorboard en http://localhost:6006/ (en mi caso), el que muestra distintos gráficos en tiempo real del entrenamiento del robot, como recompensas vs steps, duración de un episodio vs steps, etc.

Por lo que estuve viendo, debería empezar a entregar resultados a partir del step 1.000.000, y alcanza la politica "óptima" como en los 10-15 millones, asi que hay que dejarlo corriendo y tener paciencia xd.

Para probar al robot con alguno de los modelos (se puede hacer mientras esté entrenando, en otra terminal) hay que ejecutar el archivo run.py y en la linea "model" especificar el modelo a cargar (por ejemplo, "Modelos/PPO/PPO-01/best_model").

Si uno quiere parar el entrenamiento, basta con parar la ejecución del archivo. Lo bueno de esto es que como se van guardando los modelos periódicamente, se puede volver a entrenar desde donde lo dejaron.

Para resumir el entrenamiento con alguno de los métodos, tienen que comentar las siguientes líneas:
```bash
model = PPO('MlpPolicy', env, verbose=1, device='cuda' tensorboard_log=log_path)
model.learn(total_timesteps=10_000_000, callback=[eval_callback, checkpoint_callback])
```
y descomentar estas, modificando obviamente el remaining_timesteps (restando los steps del último modelo guardado):
```bash
model = PPO.load("./Modelos/PPO/PPO-01/model_640000_steps", env=env)
remaining_timesteps = 10_000_000 - 640_000
model.learn(total_timesteps=remaining_timesteps, log_interval=1, callback=[eval_callback, checkpoint_callback], reset_num_timesteps=False)
```

Para modificar los hiperparámetros de algún método, tienen que agregarlo como argumento en la línea:
```bash
model = PPO('MlpPolicy', env, verbose=1, device='cuda' tensorboard_log=log_path)
```
Por ejemplo el learning rate:
```bash
model = PPO('MlpPolicy', env, verbose=1, device='cuda' tensorboard_log=log_path, learning_rate=0.00003)
```
Cuando hice el otro proyecto solo tuvimos que modificar el learning rate, ahi cuando empiecen a entrenar al robot pueden llegar hasta cierto step y luego volver a entrenarlo con un learning rate diferente y comparar los gráficos de recompensas.
Están todos los parámetros disponibles en la documentación de stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html)

