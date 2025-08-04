import os
import torch
import carb
import gymnasium as gym
from isaaclab.envs import ManagerBasedEnv
from go2.go2_ctrl_cfg import unitree_go2_flat_cfg, unitree_go2_rough_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

base_vel_cmd_input = None

# Initialize base_vel_cmd_input as a tensor when created
def init_base_vel_cmd(num_envs):
    global base_vel_cmd_input
    base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32)

# Modify base_vel_cmd to use the tensor directly
def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
    global base_vel_cmd_input
    return base_vel_cmd_input.clone().to(env.device)


def sub_keyboard_event(event) -> bool:                                                          # [go2_ctrl.py] La fonction est déclenchée
    global base_vel_cmd_input                                                                   # [go2_ctrl.py] quand une touche du   
    lin_vel = 1.5                                                                               # [go2_ctrl.py] clavier est préssée   
    ang_vel = 1.5                                                                               # [go2_ctrl.py]                       
                                                                                                # [go2_ctrl.py]                       
    if base_vel_cmd_input is not None:                                                          # [go2_ctrl.py]                       
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:                                # [go2_ctrl.py]                       
            # Update tensor values for environment 0                                            # [go2_ctrl.py]                       
            if event.input.name == 'W':                                                         # [go2_ctrl.py]                       
                base_vel_cmd_input[0] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)      # [go2_ctrl.py]                       
            elif event.input.name == 'S':                                                       # [go2_ctrl.py]                       
                base_vel_cmd_input[0] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)     # [go2_ctrl.py]                       
            elif event.input.name == 'A':                                                       # [go2_ctrl.py]                       
                base_vel_cmd_input[0] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)      # [go2_ctrl.py]                       
            elif event.input.name == 'D':                                                       # [go2_ctrl.py]                       
                base_vel_cmd_input[0] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)     # [go2_ctrl.py]                       
            elif event.input.name == 'Z':                                                       # [go2_ctrl.py]                       
                base_vel_cmd_input[0] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)      # [go2_ctrl.py]                       
            elif event.input.name == 'C':                                                       # [go2_ctrl.py]                       
                base_vel_cmd_input[0] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)     # [go2_ctrl.py]                       
                                                                                                # [go2_ctrl.py]                       
            # If there are multiple environments, handle inputs for env 1                       # [go2_ctrl.py]                       
            if base_vel_cmd_input.shape[0] > 1:                                                 # [go2_ctrl.py]                       
                if event.input.name == 'I':                                                     # [go2_ctrl.py]                       
                    base_vel_cmd_input[1] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)  # [go2_ctrl.py]                       
                elif event.input.name == 'K':                                                   # [go2_ctrl.py]                       
                    base_vel_cmd_input[1] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32) # [go2_ctrl.py]                       
                elif event.input.name == 'J':                                                   # [go2_ctrl.py]                       
                    base_vel_cmd_input[1] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)  # [go2_ctrl.py]                       
                elif event.input.name == 'L':                                                   # [go2_ctrl.py]                       
                    base_vel_cmd_input[1] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32) # [go2_ctrl.py]                       
                elif event.input.name == 'M':                                                   # [go2_ctrl.py]                       
                    base_vel_cmd_input[1] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)  # [go2_ctrl.py]                       
                elif event.input.name == '>':                                                   # [go2_ctrl.py]                       
                    base_vel_cmd_input[1] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32) # [go2_ctrl.py]                       
                                                                                                # [go2_ctrl.py]                       
        # Reset commands to zero on key release                                                 # [go2_ctrl.py]                       
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:                            # [go2_ctrl.py]                       
            base_vel_cmd_input.zero_()                                                          # [go2_ctrl.py]                       
    return True                                                                                 # [go2_ctrl.py]                       


def get_rsl_flat_policy(cfg):                                                              # [go2_ctrl.py] Charge la policy RSL-RL flat           
    cfg.observations.policy.height_scan = None                                             # [go2_ctrl.py]                                        
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)                          # [go2_ctrl.py] Crée l'env Gym avec la config          
    env = RslRlVecEnvWrapper(env)                                                          # [go2_ctrl.py] Vectorise l'environnement              
                                                                                           # [go2_ctrl.py]                                        
    # Low level control: rsl control policy                                                # [go2_ctrl.py] Contrôle bas niveau par PPO            
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_flat_cfg                               # [go2_ctrl.py] Chargement des paramètres PPO          
                                                                                           # [go2_ctrl.py]                                        
    ckpt_path = get_checkpoint_path(                                                       # [go2_ctrl.py] Génère le chemin vers le checkpoint    
        log_path=os.path.abspath("ckpts"),                                                 # [go2_ctrl.py] Dossier des checkpoints                
        run_dir=agent_cfg["load_run"],                                                     # [go2_ctrl.py] Dossier de run à charger               
        checkpoint=agent_cfg["load_checkpoint"]                                            # [go2_ctrl.py] ID du checkpoint à charger             
    )                                                                                      # [go2_ctrl.py]                                        
                                                                                           # [go2_ctrl.py]                                        
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])  # [go2_ctrl.py] Initialise le runner PPO               
    ppo_runner.load(ckpt_path)                                                             # [go2_ctrl.py] Charge le modèle entraîné              
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])                   # [go2_ctrl.py] Extrait la policy en mode inférence    
    return env, policy                                                                     # [go2_ctrl.py] Retourne env + policy prêts à l'emploi 


def get_rsl_rough_policy(cfg):
    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_rough_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy