import mujoco
import glfw
import time

#ENTORNO: conda activate mujoco_env

# Ruta al archivo XML de Cassie
xml_path = "C:/Users/sanza/.mujoco/mujoco210/mujoco_menagerie-main/agility_cassie/scene.xml"

# Cargar el modelo y crear la simulación
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

glfw.init()  # Inicializar el contexto de renderizado GLFW

# Crear la ventana
window = glfw.create_window(800, 600, "Cassie Simulation", None, None)
glfw.make_context_current(window)

# Configurar la cámara
cam = mujoco.MjvCamera()
cam.azimuth = 120
cam.elevation = -10
cam.distance = 3.0
cam.lookat[:] = [0, 0, 1]

# Configurar la escena y el contexto de renderizado
scene = mujoco.MjvScene(model, maxgeom=1000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# Variables de tiempo y control de FPS
last_time = time.time()
frame_rate = 60  # Control de 60 FPS
sim_time_per_frame = 1.0 / frame_rate

model.opt.timestep = 0.003  # Ajusta el tiempo de paso de la simulación

# Bucle de simulación y renderizado
step_count = 0
render_freq = 3  # Renderizar solo cada 3 pasos de simulación

while not glfw.window_should_close(window):
    mujoco.mj_step(model, data)
    step_count += 1

    # Imprimir estados del robot (posiciones y velocidades)
    if step_count % render_freq == 0:
        # Obtener posiciones de las articulaciones
        joint_positions = data.qpos
        # Obtener velocidades de las articulaciones
        joint_velocities = data.qvel
        
        # Imprimir los estados
        print(f"Step {step_count}:")
        print(f"Posiciones de las articulaciones: \n{joint_positions}")
        print(f"Velocidades de las articulaciones: \n{joint_velocities}")

        # Renderizar la escena
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, 800, 600), scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
