import os
import hydra
import rclpy
import torch
import time
import math
import argparse
from isaaclab.app import AppLauncher

# [Etape] 1. Lance l'Application IsaacSim
# add argparse arguments
parser = argparse.ArgumentParser(description="Piloter le Go2 avec Ros2")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
from go2.go2_env import Go2RSLEnvCfg, camera_follow
import env.sim_env as sim_env
import go2.go2_sensors as go2_sensors
import omni
import carb
import go2.go2_ctrl as go2_ctrl
import go2.go2_env as go2_env
import ros2.go2_ros2_bridge as go2_ros2_bridge
import ros2.driver_ros2 as driver_ros2

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)


def run_simulator(cfg):

    # [Etape] 2. Charge la config de l'env GO2
    print("Etape 2")
    # Go2 Environment setup
    go2_env_cfg = Go2RSLEnvCfg()                                       # [go2_env.py] Lance la classe qui cree l'env go2
    go2_env_cfg.scene.num_envs = cfg.num_envs                          # [go2_env.py]                                   
    go2_env_cfg.decimation = math.ceil(1./go2_env_cfg.sim.dt/cfg.freq) # [go2_env.py]                                   
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation           # [go2_env.py]                                   

    # [Etape] 3. Charge la politique RSL_RL
    print("Etape 3")
    go2_env.init_base_lin_vel(1)                                       # [go2_env.py] Variable partagé pour les OBS     
    go2_env.init_base_ang_vel(1)                                       # [go2_env.py] Variable partagé pour les OBS     
    go2_env.init_quaternion(1)                                         # [go2_env.py] Variable partagé pour les OBS     
    go2_env.init_gravity(1)                                            # [go2_env.py] Variable partagé pour les OBS     
    go2_env.init_joint_pos(1)                                          # [go2_env.py] Variable partagé pour les OBS     
    go2_env.init_joint_vel(1)                                          # [go2_env.py] Variable partagé pour les OBS     
    go2_env.init_last_action(1)                                        # [go2_env.py] Variable partagé pour les OBS     
    go2_ctrl.init_base_vel_cmd(cfg.num_envs)                           # [go2_ctrl.py]Variable partagé pour les OBS     
    env, policy = go2_ctrl.get_rsl_flat_policy(go2_env_cfg)            # [go2_ctrl.py] Charge une politique             


    # [Etape] 4. Crée la scène Isaacsim
    print("Etape 4")
    # Simulation environment
    if (cfg.env_name == "obstacle-dense"):                             # [Isaaclab] Crée la scène choisie par le fichier
        sim_env.create_obstacle_dense_env() # obstacles dense          # [Isaaclab] sim_env qui utilise des fonctions   
    elif (cfg.env_name == "obstacle-medium"):                          # [Isaaclab] d'Isaaclab                          
        sim_env.create_obstacle_medium_env() # obstacles medium        # [Isaaclab]                                     
    elif (cfg.env_name == "obstacle-sparse"):                          # [Isaaclab]                                     
        sim_env.create_obstacle_sparse_env() # obstacles sparse        # [Isaaclab]                                     
    elif (cfg.env_name == "warehouse"):                                # [Isaaclab]                                     
        sim_env.create_warehouse_env() # warehouse                     # [Isaaclab]                                     
    elif (cfg.env_name == "warehouse-forklifts"):                      # [Isaaclab]                                     
        sim_env.create_warehouse_forklifts_env() # warehouse forklifts # [Isaaclab]                                     
    elif (cfg.env_name == "warehouse-shelves"):                        # [Isaaclab]                                     
        sim_env.create_warehouse_shelves_env() # warehouse shelves     # [Isaaclab]                                     
    elif (cfg.env_name == "full-warehouse"):                           # [Isaaclab]                                     
        sim_env.create_full_warehouse_env() # full warehouse           # [Isaaclab]                                     
    
    # [Etape] 5. Instancie les capteur via le fichier go2_sensors.py
    print("Etape 5")
    # Sensor setup
    sm = go2_sensors.SensorManager(cfg.num_envs)   # [Isaaclab] Ajoute les capteurs à la simulation
    lidar_annotators = sm.add_rtx_lidar()          # [Isaaclab]                                    
    cameras = sm.add_camera(cfg.freq)              # [Isaaclab]                                    

    # [Etape] 6. Connecte le clavier de l'interface d'ILab
    print("Etape 6")
    # Keyboard control
    #system_input = carb.input.acquire_input_interface()                                      # [go2_ctrl.py] Abonne le clavier a une fonction
    #system_input.subscribe_to_keyboard_events(                                               # [go2_ctrl.py] de go2_ctrl.py qui est          
    #    omni.appwindow.get_default_app_window().get_keyboard(), go2_ctrl.sub_keyboard_event) # [go2_ctrl.py] base_vel_cmd pour la modifier   
    
    # [Etape] 7. Initialisation de ROS2
    print("Etape 7")
    go2_env_cfg.scene.num_envs = 1  

    # ROS2 Bridge
    rclpy.init()                                                               # [ROS2] Lance ROS2 et initialise le node RobotDataManager qui récupère

    dm = go2_ros2_bridge.RobotDataManager(env, lidar_annotators, cameras, cfg) # [ROS2]  
    ol = driver_ros2.OdomListener()                                            # [ROS2]  
    cl = driver_ros2.CmdListener()                                             # [ROS2]  
    qt = driver_ros2.QuaternionListener()                                      # [ROS2]  
    jl = driver_ros2.JointListener()                                           # [ROS2]  

    from rclpy.executors import MultiThreadedExecutor           # [ROS2] La partie la gère le multithread pour les 
    import threading                                            # [ROS2] nodes RDM et driver_ros2                  
    # ➕ Utilise un MultiThreadedExecutor                       # [ROS2]                                           
    executor = MultiThreadedExecutor()                          # [ROS2]                                           
    executor.add_node(dm)                                       # [ROS2]                                           
    executor.add_node(ol)                                       # [ROS2]                                           
    executor.add_node(cl)                                       # [ROS2]                                           
    executor.add_node(qt)                                       # [ROS2]                                           
    executor.add_node(jl)                                       # [ROS2]                                           

                                                                # [ROS2]                                           
    # Lance l'exécuteur dans un thread                          # [ROS2]                                           
    threading.Thread(target=executor.spin, daemon=True).start() # [ROS2]                                           

    obs_isaacsim_log = []
    obs_ros_log = []
    start_time_global = time.time()


    # [Etape] 8. Lancement de la boucle principale de Simulation
    # Run simulation
    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
    obs, _ = env.reset()
    dm.pub_ros2_data()
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():   
            # control joints
            actions = policy(obs)  # [go2_env.py] Lance le calcul des observations contenus dans go2_env.py
            go2_env.last_action_input[0] = actions[0].detach().cpu()

            # step the environment
            obs, _, _, _ = env.step(actions)

            # # ROS2 data     
            dm.pub_ros2_data()   # [ROS2] Lance les fonctions de publications de ROS2
            #rclpy.spin_once(dm) # [ROS2]                                            
            #rclpy.spin_once(ol) # [ROS2]                                            

            asset = env.unwrapped.scene["unitree_go2"]
            #print("Default vel:")
            #print(asset.data.default_joint_vel)
            quat_sim = asset.data.body_link_quat_w[0].cpu().tolist()


            # Stocke Valeur IsaacSim                                                  # [Etape] Stocke la variable venant
            obs_isaacsim_log.append(obs[0][36:48].cpu().tolist())                     # [Etape] de Isaacsim              
                                                                                      # [Etape]                          
            # Stocke Valeur ROS                                                       # [Etape] Stocke la variable venant
            obs_ros_log.append(go2_env.last_action_input[0].clone().cpu().tolist())   # [Etape] de Ros2                  

            
            # Camera follow
            if (cfg.camera_follow):
                camera_follow(env)

            # limit loop time
            elapsed_time = time.time() - start_time
            if elapsed_time < sim_step_dt:
                sleep_duration = sim_step_dt - elapsed_time
                time.sleep(sleep_duration)
        actual_loop_time = time.time() - start_time
        rtf = min(1.0, sim_step_dt/elapsed_time)
        print(f"\rStep time: {actual_loop_time*1000:.2f}ms, Real Time Factor: {rtf:.2f}", end='', flush=True)
        # Arrêt après 10 secondes
        if time.time() - start_time_global > 100.0:
            break



    """        
    # Transposer pour obtenir x(t), y(t), z(t)
    obs_0, obs_1, obs_2, obs_3, obs_4, obs_5, obs_6, obs_7, obs_8, obs_9, obs_10, obs_11 = zip(*obs_isaacsim_log)
    ros_0, ros_1, ros_2, ros_3, ros_4, ros_5, ros_6, ros_7, ros_8, ros_9, ros_10, ros_11= zip(*obs_ros_log)
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_vel(title, obs_vals, ros_vals, ylabel):
        num_points = len(obs_vals)
        x_vals = list(range(num_points))
        xtick_positions = x_vals[::3]  # 1 tick sur 3

        plt.figure()
        plt.plot(x_vals, obs_vals, label="IsaacSim", marker='o')
        plt.plot(x_vals, ros_vals, label="ROS2", marker='x', linestyle='--')
        plt.xticks(xtick_positions)  # Affiche les ticks tous les 3 points
        plt.title(title)
        plt.xlabel("Étape")
        plt.ylabel(ylabel)
        plt.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()

    test0 = np.array(obs_0) - np.array(ros_0)
    test1 = np.array(obs_1) - np.array(ros_1)
    test2 = np.array(obs_2) - np.array(ros_2)
    test3 = np.array(obs_3) - np.array(ros_3)
    test4 = np.array(obs_4) - np.array(ros_4)
    test5 = np.array(obs_5) - np.array(ros_5)
    test6 = np.array(obs_6) - np.array(ros_6)
    test7 = np.array(obs_7) - np.array(ros_7)
    test8 = np.array(obs_8) - np.array(ros_8)
    test9 = np.array(obs_9) - np.array(ros_9)
    test10 = np.array(obs_10) - np.array(ros_10)
    test11 = np.array(obs_11) - np.array(ros_11)
    plt.plot(test0, label="Patte 0")
    plt.plot(test1, label="Patte 1")
    plt.plot(test2, label="Patte 2")
    plt.plot(test3, label="Patte 3")
    plt.plot(test4, label="Patte 4")
    plt.plot(test5, label="Patte 5")
    plt.plot(test6, label="Patte 6")
    plt.plot(test7, label="Patte 7")
    plt.plot(test8, label="Patte 8")
    plt.plot(test9, label="Patte 9")
    plt.plot(test10, label="Patte 10")
    plt.plot(test11, label="Patte 11")
    plt.legend()

    plot_vel("Patte 0", obs_0, ros_0, "Patte 0")
    plot_vel("Patte 1", obs_1, ros_1, "Patte 1")
    plot_vel("Patte 2", obs_2, ros_2, "Patte 2")
    plot_vel("Patte 3", obs_3, ros_3, "Patte 3")
    plot_vel("Patte 4", obs_4, ros_4, "Patte 4")
    plot_vel("Patte 5", obs_5, ros_5, "Patte 5")
    plot_vel("Patte 6", obs_6, ros_6, "Patte 6")
    plot_vel("Patte 7", obs_7, ros_7, "Patte 7")
    plot_vel("Patte 8", obs_8, ros_8, "Patte 8")
    plot_vel("Patte 9", obs_9, ros_9, "Patte 9")
    plot_vel("Patte 10", obs_10, ros_10, "Patte 10")
    plot_vel("Patte 11", obs_11, ros_11, "Patte 11")

    plt.show()
    """

    ol.destroy_node()
    dm.destroy_node()
    rclpy.shutdown()
    executor.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    run_simulator()
    