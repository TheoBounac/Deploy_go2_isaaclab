from isaaclab.scene import InteractiveSceneCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import UniformNoiseCfg
from isaacsim.core.utils.viewports import set_camera_view
import numpy as np
from scipy.spatial.transform import Rotation as R
import go2.go2_ctrl as go2_ctrl
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import quat_apply_inverse


@configclass
class Go2SimCfg(InteractiveSceneCfg):                                                       # [Isaaclab]                                
    # ground plane                                                                          # [Isaaclab] Configuration de la scène de   
    ground = AssetBaseCfg(                                                                  # [Isaaclab] Isaaclab                       
        prim_path="/World/ground",                                                          # [Isaaclab]                                
        spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300.0, 300.0), physics_material=sim_utils.RigidBodyMaterialCfg( static_friction=1, dynamic_friction=1, friction_combine_mode="multiply",restitution_combine_mode="multiply")),         
        init_state=AssetBaseCfg.InitialStateCfg(                                            # [Isaaclab]                                
            pos=(0, 0, 1e-4)                                                                # [Isaaclab]                                
        )                                                                                   # [Isaaclab]                                
    )                                                                                       # [Isaaclab]                                
                                                                                            # [Isaaclab]                                
    # lights                                                                                # [Isaaclab]                                
    # Lights                                                                                # [Isaaclab]                                
    light = AssetBaseCfg(                                                                   # [Isaaclab]                                
        prim_path="/World/Light",                                                           # [Isaaclab]                                
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),        # [Isaaclab]                                
    )                                                                                       # [Isaaclab]                                
    sky_light = AssetBaseCfg(                                                               # [Isaaclab]                                
        prim_path="/World/DomeLight",                                                       # [Isaaclab]                                
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),               # [Isaaclab]                                
    )                                                                                       # [Isaaclab]                                
    # dome_light = AssetBaseCfg(                                                            # [Isaaclab]                                
    #     prim_path="/World/DomeLight",                                                     # [Isaaclab]                                
    #     spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),             # [Isaaclab]                                
    # )                                                                                     # [Isaaclab]                                
                                                                                            # [Isaaclab]                                
    # Go2 Robot                                                                             # [Isaaclab]                                
    unitree_go2: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Go2")  # [Isaaclab] C'est le fichier d'articulation
    # Go2 foot contact sensor                                                               # [Isaaclab] du GO2                         
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Go2/.*_foot", history_length=3, track_air_time=True)  # [Isaaclab]      
                                                                                                                      # [Isaaclab]      
    # Go2 height scanner                                                                                              # [Isaaclab]      
    height_scanner = RayCasterCfg(                                                                                    # [Isaaclab]      
        prim_path="{ENV_REGEX_NS}/Go2/base",                                                                          # [Isaaclab]      
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20)),                                                            # [Isaaclab]      
        attach_yaw_only=True,                                                                                         # [Isaaclab]      
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),                                         # [Isaaclab]      
        debug_vis=False,                                                                                              # [Isaaclab]      
        mesh_prim_paths=["/World/ground"],                                                                            # [Isaaclab]      
    )

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="unitree_go2", joint_names=[".*"])


base_lin_vel_input = None                                                        # [ROS2] Buffer partagé pour l'observation base_lin_vel issue de ROS 
                                                                                 # [ROS2] 
# Initialize base_lin_vel_input as a tensor when created                         # [ROS2] 
def init_base_lin_vel(num_envs):                                                 # [ROS2] 
    global base_lin_vel_input                                                    # [ROS2] 
    base_lin_vel_input = torch.zeros((num_envs, 3), dtype=torch.float32)         # [ROS2] 
                                                                                 # [ROS2] 
# Modify base_vel_cmd to use the tensor directly                                 # [ROS2] 
def base_lin_vel(env: ManagerBasedEnv) -> torch.Tensor:                          # [ROS2] 
    global base_lin_vel_input                                                    # [ROS2] 
    return base_lin_vel_input.clone().to(env.device)                             # [ROS2] 

base_ang_vel_input = None                                                        # [ROS2] Buffer partagé pour l'observation base_ang_vel issue de ROS 
                                                                                 # [ROS2] 
# Initialize base_lin_vel_input as a tensor when created                         # [ROS2] 
def init_base_ang_vel(num_envs):                                                 # [ROS2] 
    global base_ang_vel_input                                                    # [ROS2] 
    base_ang_vel_input = torch.zeros((num_envs, 3), dtype=torch.float32)         # [ROS2] 
                                                                                 # [ROS2] 
# Modify base_vel_cmd to use the tensor directly                                 # [ROS2] 
def base_ang_vel(env: ManagerBasedEnv) -> torch.Tensor:                          # [ROS2] 
    global base_ang_vel_input                                                    # [ROS2] 
    return base_ang_vel_input.clone().to(env.device)                             # [ROS2] 

gravity_input = None                                                             # [ROS2] Buffer partagé pour l'observation Gravity issue de ROS 
                                                                                 # [ROS2] 
# Initialize gravity_orientation_input as a tensor when created                  # [ROS2] 
def init_gravity(num_envs):                                                      # [ROS2] 
    global gravity_input                                                         # [ROS2] 
    gravity_input = torch.zeros((num_envs, 3), dtype=torch.float32)              # [ROS2] 
                                                                                 # [ROS2] 
def gravity(env: ManagerBasedEnv) -> torch.Tensor:                               # [ROS2] 
    global gravity_input                                                         # [ROS2] 
    return gravity_input.clone().to(env.device)                                  # [ROS2] 

joint_pos_input = None                                                           # [ROS2]  Buffer partagé pour l'observation Joint_pos issue de ROS 
                                                                                 # [ROS2] 
# Initialize gjoint_pos_input as a tensor when created                           # [ROS2] 
def init_joint_pos(num_envs):                                                    # [ROS2] 
    global joint_pos_input                                                       # [ROS2] 
    joint_pos_input = torch.zeros((num_envs, 12), dtype=torch.float32)           # [ROS2] 
                                                                                 # [ROS2] 
def joint_pos(env: ManagerBasedEnv) -> torch.Tensor:                             # [ROS2] 
    global joint_pos_input                                                       # [ROS2] 
    return joint_pos_input.clone().to(env.device)                                # [ROS2] 


joint_vel_input = None                                                           # [ROS2] Buffer partagé pour l'observation Joint_vel issue de ROS 
                                                                                 # [ROS2] 
# Initialize gjoint_pos_input as a tensor when created                           # [ROS2] 
def init_joint_vel(num_envs):                                                    # [ROS2] 
    global joint_vel_input                                                       # [ROS2] 
    joint_vel_input = torch.zeros((num_envs, 12), dtype=torch.float32)           # [ROS2] 
                                                                                 # [ROS2] 
def joint_vel(env: ManagerBasedEnv) -> torch.Tensor:                             # [ROS2] 
    global joint_vel_input                                                       # [ROS2] 
    return joint_vel_input.clone().to(env.device)                                # [ROS2] 


last_action_input =None                                                          #[Etape] Buffer partagé pour l'observation action issue de policy
                                                                                 #[Etape] 
def init_last_action(num_envs):                                                  #[Etape] 
    global last_action_input                                                     #[Etape] 
    last_action_input = torch.zeros((num_envs, 12), dtype=torch.float32)         #[Etape] 
                                                                                 #[Etape] 
def last_action(env: ManagerBasedEnv) -> torch.Tensor:                           #[Etape] 
    global last_action_input                                                     #[Etape] 
    return last_action_input.clone().to(env.device)                              #[Etape] 

quaternion_input = None                                                                # [ROS2] Buffer partagé pour gravity_orientation issue de ROS 
                                                                                       # [ROS2] 
# Initialize gravity_orientation_input as a tensor when created                        # [ROS2] 
def init_quaternion(num_envs):                                                         # [ROS2] 
    global quaternion_input                                                            # [ROS2] 
    quaternion_input = torch.zeros((num_envs, 4), dtype=torch.float32)                 # [ROS2] 
                                                                                       # [ROS2] 
# Modify base_vel_cmd to use the tensor directly                                       # [ROS2] 
def quaternion(env: ManagerBasedEnv) -> torch.Tensor:                                  # [ROS2] 
    global quaternion_input                                                            # [ROS2] 
    return quaternion_input.clone().to(env.device)                                     # [ROS2] 
                                                                                       # [ROS2] 
                                                                                       # [ROS2] 
def get_projected_gravity_from_quaternion(quat_tensor):                                # [ROS2] 
    # quat_tensor : [w, x, y, z] (1D Tensor ou liste)                                  # [ROS2] 
    qw = quat_tensor[0]                                                                # [ROS2] 
    qx = quat_tensor[1]                                                                # [ROS2] 
    qy = quat_tensor[2]                                                                # [ROS2] 
    qz = quat_tensor[3]                                                                # [ROS2] 
                                                                                       # [ROS2] 
    gravity = torch.zeros(3)                                                           # [ROS2] 
                                                                                       # [ROS2] 
    # Applique rotation inverse de [0, 0, -1] dans le repère du robot                  # [ROS2] 
    # Formule dérivée de R^T * [0, 0, -1] où R est la matrice de rotation du quaternion# [ROS2] 
                                                                                       # [ROS2] 
    gravity[0] = 2 * (-qx * qz + qw * qy)                                              # [ROS2] 
    gravity[1] = -2 * (qy * qz + qw * qx)                                              # [ROS2] 
    gravity[2] = -(1 - 2 * (qx**2 + qy**2))                                            # [ROS2] 
                                                                                       # [ROS2] 
    return gravity                                                                     # [ROS2] 
                                                                                       # [ROS2] 
def get_projected_gravity_from_quaternion_2(quat_tensor):                              # [ROS2] 
    quat = quat_tensor.unsqueeze(0)  # shape (1, 4)                                    # [ROS2] 
    gravity_world = torch.tensor([[0.0, 0.0, -1.0]])  # shape (1, 3)                   # [ROS2] 
                                                                                       # [ROS2] 
    g_proj = quat_apply_inverse(quat, gravity_world)  # shape (1, 3)                   # [ROS2] 
    gravity = torch.zeros(3)                                                           # [ROS2] 
                                                                                       # [ROS2] 
    projected_gravity = g_proj[0]  # shape (3,)                                        # [ROS2] 
                                                                                       # [ROS2] 
    return projected_gravity                                                           # [ROS2] 


                                                        



@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        #base_lin_vel = ObsTerm(func=mdp.base_lin_vel,                                        # [Observations] Les fonctions d'observation sont 
        #                     params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})       # [Observations] notées ici                       
        base_lin_vel = ObsTerm(func=base_lin_vel)                                             # [Observations]                                  

        #base_ang_vel = ObsTerm(func=mdp.base_ang_vel,                                        # [Observations]                                  
        #                       params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})     # [Observations]                                  
        base_ang_vel = ObsTerm(func=base_ang_vel)                                             # [Observations]                                  

        #projected_gravity = ObsTerm(func=mdp.projected_gravity,                              # [Observations]                                  
        #                            params={"asset_cfg": SceneEntityCfg(name="unitree_go2")},# [Observations]                                  
        #                           noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))           # [Observations]                                  
        projected_gravity = ObsTerm(func=gravity)                                             # [Observations]                                  

        # velocity command                                                                   # [Observations]                                  
        base_vel_cmd = ObsTerm(func=go2_ctrl.base_vel_cmd)                                   # [go2_ctrl.py] Regarde la valeur stockées dans   
                                                                                             # [go2_ctrl.py] go2_ctrl.py                       
        
        #joint_pos = ObsTerm(func=mdp.joint_pos_rel,                                         # [Observations]                                  
        #                    params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})       # [Observations]                                  
        joint_pos = ObsTerm(func=joint_pos)                                                  # [Observations]                                  
        
        
        #joint_vel = ObsTerm(func=mdp.joint_vel_rel,                                         # [Observations]                                  
        #                    params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})       # [Observations]                                  
        joint_vel = ObsTerm(func = joint_vel)                                                # [Observations]                                  
        
        #actions = ObsTerm(func=mdp.last_action)                                             # [Observations]                                  
        actions = ObsTerm(func=last_action)                                                  # [Observations]                                  
                                                                                             
        # Height scan                                                                        # [Observations]                                  
        height_scan = ObsTerm(func=mdp.height_scan,                                          # [Observations]                                  
                              params={"sensor_cfg": SceneEntityCfg("height_scanner")},       # [Observations]                                  
                              clip=(-1.0, 1.0))                                              # [Observations]                                  
                                                                                             # [Observations]                                  
        def __post_init__(self) -> None:                                                     # [Observations]                                  
            self.enable_corruption = False                                                   # [Observations]                                  
            self.concatenate_terms = True                                                    # [Observations]                                  

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_vel_cmd = mdp.UniformVelocityCommandCfg(
        asset_name="unitree_go2",
        resampling_time_range=(0.0, 0.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )

@configclass
class EventCfg:
    """Configuration for events."""
    pass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    pass

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass

@configclass
class Go2RSLEnvCfg(ManagerBasedRLEnvCfg):                                              # [go2_env.py] C'est la classe qui définit l'env    
    """Configuration for the Go2 environment."""                                       # [go2_env.py] pour le GO2                          
    # scene settings                                                                   # [Isaaclab] La scène Isaacsim                      
    scene = Go2SimCfg(num_envs=2, env_spacing=2.0)                                     # [Isaaclab]                                        

    # basic settings                                                                   # [Observations] Observation et action              
    observations = ObservationsCfg()                                                   # [go2_env.py] Observations pour la policy          
    actions = ActionsCfg()                                                             # [go2_env.py] Actions envoyées à la policy         
                                                                                        # [go2_env.py]                                     
    # dummy settings                                                                   # [go2_env.py] Modules vides à remplir              
    commands = CommandsCfg()                                                           # [go2_env.py] Commandes de déplacement (désactivées)
    rewards = RewardsCfg()                                                             # [go2_env.py] Récompenses MDP (à définir)          
    terminations = TerminationsCfg()                                                   # [go2_env.py] Conditions de fin d'épisode          
    events = EventCfg()                                                                # [go2_env.py] Événements personnalisés             
    curriculum = CurriculumCfg()                                                       # [go2_env.py] Apprentissage progressif (non utilisé)

    def __post_init__(self):                                                           # [go2_env.py] Initialisation post-config           
        # viewer settings                                                              # [Isaaclab] Caméra suiveuse                        
        self.viewer.eye = [-4.0, 0.0, 5.0]                                             # [Isaaclab] Position de la caméra                  
        self.viewer.lookat = [0.0, 0.0, 0.0]                                           # [Isaaclab] Cible de la caméra                     

        # step settings                                                                # [Isaaclab] Pas de contrôle                        
        self.decimation = 8                                                            # [Isaaclab] Nombre de steps par action             

        # simulation settings                                                          # [Isaaclab] Paramètres de simulation physique      
        self.sim.dt = 0.005                                                            # [Isaaclab] Pas de temps de la simulation          
        self.sim.render_interval = self.decimation                                     # [Isaaclab] Fréquence de rendu visuel              
        self.sim.disable_contact_processing = True                                     # [Isaaclab] Désactive le traitement de contact     
        self.sim.render.antialiasing_mode = None                                       # [Isaaclab] Pas d'antialiasing (gain perf)         
        # self.sim.physics_material = self.scene.terrain.physics_material              # [Isaaclab] Matériau de physique du terrain        

        # settings for rsl env control                                                 # [go2_env.py] Paramètres pour l'entraînement              
        self.episode_length_s = 20.0                                                   # [go2_env.py] Longueur d’épisode (inutile ici)            
        self.is_finite_horizon = False                                                 # [go2_env.py] Horizon infini pour les épisodes            
        self.actions.joint_pos.scale = 0.25                                            # [go2_env.py] Échelle des actions (clamp du policy output)

        if self.scene.height_scanner is not None:                                      # [go2_env.py] Si scanner actif → fréquence mise à jour   
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt    # [go2_env.py] Fréquence = intervalle entre steps         
 



def camera_follow(env):
    if (env.unwrapped.scene.num_envs == 1):
        robot_position = env.unwrapped.scene["unitree_go2"].data.root_state_w[0, :3].cpu().numpy()
        robot_orientation = env.unwrapped.scene["unitree_go2"].data.root_state_w[0, 3:7].cpu().numpy()
        rotation = R.from_quat([robot_orientation[1], robot_orientation[2], 
                                robot_orientation[3], robot_orientation[0]])
        yaw = rotation.as_euler('zyx')[0]
        yaw_rotation = R.from_euler('z', yaw).as_matrix()
        set_camera_view(
            yaw_rotation.dot(np.asarray([-4.0, 0.0, 5.0])) + robot_position,
            robot_position
        )