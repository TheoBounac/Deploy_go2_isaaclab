from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch 
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import *

###### Pour diriger le path vers la ou es stocke unitree_sd2k_python ######
sys.path.append('.')
sys.path.append('..')
sys.path.append(LEGGED_GYM_ROOT_DIR+'/..')
###########################################################################

from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2_python.unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2_python.unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2_python.unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2_python.unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2_python.unitree_sdk2py.utils.crc import CRC
from unitree_sdk2_python.unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2_python.unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

from deploy.deploy_real.common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from deploy.deploy_real.common.rotation_helper import get_gravity_orientation, transform_imu_data
from deploy.deploy_real.common.remote_controller import RemoteController, KeyMap
from deploy.deploy_real.config import Config


class Controller:
    def __init__(self, config: Config) -> None:
        # Fichier de config pour le go2
        self.config = config

        # Initialisation du controller
        self.remote_controller = RemoteController()

        # Initialisation du modele
        self.policy = torch.jit.load(config.policy_path)
     
        # Initialisation des variables pour le demarrage 
        self.cmd = np.array([0, 0, 0])
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.base_lin_vel = np.array([0, 0, 0])

        # Donnees pour lever le robot en default pos
        self.dt = 0.002  
        self.startPos = [0.0] * 12
        self.duration_1 = 500
        self.duration_2 = 500
        self.duration_3 = 1000
        self.duration_4 = 900
        self.percent_1 = 0
        self.percent_2 = 0
        self.percent_3 = 0
        self.percent_4 = 0
        self.firstRun = True
        self.counter = 0

        # Positions cibles pour lever le go2
        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        self._targetPos_2 = self.config.default_angles
        self._targetPos_3 = self.config.default_angles

        # Données pour le calculateur de vitesse (plus window_size est grand plus le calcul de vitesse est lisse)
        window_size = 40
        self.vx_window = [0] * window_size
        self.vy_window = [0] * window_size
        self.vz_window = [0] * window_size

        # Initialisation des channels
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        self.sportstate_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.sportstate_subscriber.Init(self.SportStateMessageHandler, 10)

        # Initialisation des messages pour CMD et STATE
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        # Wait for the subscriber to receive data
        self.wait_for_low_state()
        
        # Initialize the commands of the motors
        init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)
    


    # Init est appelé au debut pour passer en Bas Niveau
    def Init(self):
        
        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        
        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()        # Avant de commencer on met le robot en position allonge
            self.msc.ReleaseMode()     # On libere le mode normal
            print("Le robot est en position allongé et le mode Haut niveau est relâché -> Passage en bas niveau")
            status, result = self.msc.CheckMode()
            time.sleep(1)
        
    # Fonction qui vérifie si le message low_state est bien recu    
    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Connecté au robot")


    # Fonctions Handler sont appeles des que le msg en parametre est recu dans un canal
    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def SportStateMessageHandler(self, sport_state_msg):
        self.velocity = sport_state_msg.velocity  # Récupération de la vitesse
        print("SPORT")


    # Fonction qui envoi une commande mise en paramètre au robot
    def send_cmd(self, cmd: LowCmdGo):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    # Fonction qui envoit une commande nulle en attendant que le bouton start soit appuye ce qui declenchera la levée du robot
    def zero_torque_state(self):
        print("Zero torque mode")
        print("En attente du bouton start...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    # Fonction qui lève le robot en position debout et qui le maintient tant que le bouton "A" n'est pas apuye ce qui lancera le modele
    def move_to_default_pos(self):
        print("Le robot se déplace vers la DEFAULT POSE")
        
        # Donnees
        dof_idx = self.config.leg_joint2motor_idx 
        dof_size = len(dof_idx)
        done = False
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
    
        if self.firstRun:
            for i in range(12):
                self.startPos[i] = self.low_state.motor_state[i].q
            self.firstRun = False
        self.count = 0
        while not done:
            self.count += 1

            if self.firstRun:
                for i in range(12):
                    self.startPos[i] = self.low_state.motor_state[i].q
                self.firstRun = False

            self.percent_1 += 1.0 / self.duration_1
            self.percent_1 = min(self.percent_1, 1)
            if self.percent_1 < 1:
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * self._targetPos_1[i]
                    self.low_cmd.motor_cmd[i].dq = 0
                    self.low_cmd.motor_cmd[i].kp =60
                    self.low_cmd.motor_cmd[i].kd = 5
                    self.low_cmd.motor_cmd[i].tau = 0

            if (self.percent_1 == 1) and (self.percent_2 <= 1):
                self.percent_2 += 1.0 / self.duration_2
                self.percent_2 = min(self.percent_2, 1)
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = (1 - self.percent_2) * self._targetPos_1[i] + self.percent_2 * self._targetPos_2[i]
                    self.low_cmd.motor_cmd[i].dq = 0
                    self.low_cmd.motor_cmd[i].kp = 60
                    self.low_cmd.motor_cmd[i].kd = 5
                    self.low_cmd.motor_cmd[i].tau = 0

            if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 < 1):
                self.percent_3 += 1.0 / self.duration_3
                self.percent_3 = min(self.percent_3, 1)
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = self._targetPos_2[i] 
                    self.low_cmd.motor_cmd[i].dq = 0
                    self.low_cmd.motor_cmd[i].kp = 60
                    self.low_cmd.motor_cmd[i].kd = 5
                    self.low_cmd.motor_cmd[i].tau = 0

            if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 == 1) and (self.percent_4 <= 1):
                self.percent_4 += 1.0 / self.duration_4
                self.percent_4 = min(self.percent_4, 1)
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = (1 - self.percent_4) * self._targetPos_2[i] + self.percent_4 * self._targetPos_3[i]
                    self.low_cmd.motor_cmd[i].dq = 0
                    self.low_cmd.motor_cmd[i].kp = 60
                    self.low_cmd.motor_cmd[i].kd = 5
                    self.low_cmd.motor_cmd[i].tau = 0

            self.send_cmd(self.low_cmd)
            if self.percent_4 == 1.0 or self.count == 2500000000:
                done = True
            time.sleep(0.001)
        print("LE ROBOT SE MAINTIENT DEBOUT. APPUYEZ SUR 'A' POUR DEMARRER LE MODELE")
        while self.remote_controller.button[KeyMap.A] != 1:
            default = self.config.default_angles
            for i in range(12):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = default[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = 60
                self.low_cmd.motor_cmd[motor_idx].kd = 5
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                self.send_cmd(self.low_cmd)
                time.sleep(0.002)


    # Fonction qui va allonger le robot depuis un etat debout
    def move_to_ground(self):
        percent = 0
        pos_init=[]
        for k in range(12):
            pos_init.append(self.low_state.motor_state[k].q)
        while percent != 1:
            percent += 1.0 / 300
            percent = min(percent, 1)
            couche = [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65, -0.5, 1.36, -2.65, 0.5, 1.36, -2.65]
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - percent) * pos_init[i] + percent * couche[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = 60
                self.low_cmd.motor_cmd[i].kd = 5
                self.low_cmd.motor_cmd[i].tau = 0
                self.send_cmd(self.low_cmd)
            time.sleep(0.002)
        print("LE ROBOT EST ALLONGE")

    # Fonction qui va calculer la vitesse instantanée du robot à partir des données des joints du robot
    def compute_velocity(self,theta1,theta2,theta3,thetav1,thetav2,thetav3,foot):
        vitessex = 0
        vitessey = 0
        vitessez = 0
        l1 = 0.21
        l2 = 0.23
        num_pate = 0
        for k in range(4):
            if foot[k] < 20:
                pate_au_sol = 0
            else:
                pate_au_sol = 1
                num_pate += 1
            vitessex += pate_au_sol * ((-l1*sin(theta1[k])-l2*sin(theta1[k]+theta2[k]))*-thetav1[k] + (-l2*sin(theta1[k]+theta2[k]))*-thetav2[k])
            vitessey += pate_au_sol * ( (l1*cos(theta1[k])+l2*cos(theta1[k]+theta2[k]))*sin(theta3[k])*-thetav1[k]   +   (l2*cos(theta1[k]+theta2[k])*sin(theta3[k]))*-thetav2[k]  +  (l1*sin(theta1[k])+l2*sin(theta1[k]+theta2[k]))*cos(theta3[k])*thetav3[k])   
            vitessez += pate_au_sol * ( -l1*cos(theta1[k])*thetav1[k] -l2*cos(theta1[k]+theta2[k])*(thetav1[k]+thetav2[k])) 
        if num_pate > 0:
            vitessex = vitessex / num_pate
            vitessey = vitessey / num_pate
            vitessez = vitessez / num_pate
        return vitessex, vitessey, vitessez




    # Fonction principale qui tourne en boucle lorsque le modèle tourne
    def run(self):
        self.counter += 1

        ###### Recalibrage des données pour le calcul des vitesses #######
        theta1 = []
        theta2 = []
        theta3 = []
        thetav1 = []
        thetav2 = []
        thetav3 = []
        theta1.append(-self.low_state.motor_state[1].q + 1.5708)
        theta2.append(-self.low_state.motor_state[2].q - 1.7 + pi/2)
        theta3.append(-self.low_state.motor_state[0].q)
        thetav1.append(self.low_state.motor_state[1].dq)
        thetav2.append(self.low_state.motor_state[2].dq)
        thetav3.append(-self.low_state.motor_state[0].dq)

        theta1.append(-self.low_state.motor_state[4].q + 1.5708)
        theta2.append(-self.low_state.motor_state[5].q - 1.7 + pi/2)
        theta3.append(-self.low_state.motor_state[3].q)
        thetav1.append(self.low_state.motor_state[4].dq)
        thetav2.append(self.low_state.motor_state[5].dq)
        thetav3.append(-self.low_state.motor_state[3].dq)

        theta1.append(-self.low_state.motor_state[7].q + 1.5708)
        theta2.append(-self.low_state.motor_state[8].q - 1.7 + pi/2)
        theta3.append(-self.low_state.motor_state[6].q)
        thetav1.append(self.low_state.motor_state[7].dq)
        thetav2.append(self.low_state.motor_state[8].dq)
        thetav3.append(-self.low_state.motor_state[6].dq)

        theta1.append(-self.low_state.motor_state[10].q + 1.5708)
        theta2.append(-self.low_state.motor_state[11].q - 1.7 + pi/2)
        theta3.append(-self.low_state.motor_state[9].q)
        thetav1.append(self.low_state.motor_state[10].dq)
        thetav2.append(self.low_state.motor_state[11].dq)
        thetav3.append(-self.low_state.motor_state[9].dq)
        
        foot = self.low_state.foot_force
        
        ############################ Vx #####################################################
        vx_calc = self.compute_velocity(theta1,theta2,theta3,thetav1,thetav2,thetav3,foot)[0]
        temp = 0
        for k in range(len(self.vx_window)):
            temp += self.vx_window[k]
        temp += vx_calc*5
        temp = temp/(len(self.vx_window)+5)
        
        for k in range(len(self.vx_window)-1):
            self.vx_window[k] = self.vx_window[k+1]
        self.vx_window[len(self.vx_window)-1] = temp
        vx = temp

        ############################ Vy #####################################################
        vy_calc = self.compute_velocity(theta1,theta2,theta3,thetav1,thetav2,thetav3,foot)[1]
        temp = 0
        for k in range(len(self.vy_window)):
            temp += self.vy_window[k]
        temp += vy_calc*5
        temp = temp/(len(self.vy_window)+5)
        
        for k in range(len(self.vy_window)-1):
            self.vy_window[k] = self.vy_window[k+1]
        self.vy_window[len(self.vy_window)-1] = temp
        vy = temp

        ############################ Vz #####################################################
        vz_calc = self.compute_velocity(theta1,theta2,theta3,thetav1,thetav2,thetav3,foot)[2]
        temp = 0
        for k in range(len(self.vz_window)):
            temp += self.vz_window[k]
        temp += vz_calc*5
        temp = temp/(len(self.vz_window)+5)
        
        for k in range(len(self.vz_window)-1):
            self.vz_window[k] = self.vz_window[k+1]
        self.vz_window[len(self.vz_window)-1] = temp
        vz = temp




        w = self.low_state.imu_state.gyroscope[2]

        # Récupérer les positions et vitesses des joints dans l'ordre du modèle
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        count = self.counter * self.config.control_dt

        self.cmd[0] = self.remote_controller.ly 
        self.cmd[1] = self.remote_controller.lx * -1 
        self.cmd[2] = self.remote_controller.rx * -1

        num_actions = self.config.num_actions

        # Les entrees du reseau de neurones :
        self.obs[:3]= [vx*2,vy*2,vz*2] 
        self.obs[3:6] = ang_vel * 0.25
        self.obs[6:9] = gravity_orientation
        self.obs[9:12] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.obs[12 : 12 + num_actions] = qj_obs
        self.obs[12 + num_actions : 12 + num_actions * 2] = dqj_obs
        self.obs[12 + num_actions * 2 : 12 + num_actions * 3] = self.action

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        a = self.cmd[0]
        b = self.cmd[1]
        c = self.cmd[2]
        d = self.obs[9]
        e = self.obs[10]
        f = self.obs[11]
        print(f'\r{a,b,c,d}', end='', flush=True)

        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = 35 # normalement: self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = 2.5 # normalement: self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        
        # Pour visualiser les torques de la première patte
        torque_0 = (target_dof_pos[0]-self.low_state.motor_state[3].q)*20 - self.low_state.motor_state[3].dq*2
        torque_1 = (target_dof_pos[1]-self.low_state.motor_state[4].q)*20 - self.low_state.motor_state[4].dq*2
        torque_2 = (target_dof_pos[2]-self.low_state.motor_state[5].q)*20 - self.low_state.motor_state[5].dq*2
        q_obs = qj_obs

        # send the command
        self.send_cmd(self.low_cmd)
        b0 = round(self.low_cmd.motor_cmd[0].q,2)
        b1 = round(self.low_cmd.motor_cmd[1].q,2)
        b2 = round(self.low_cmd.motor_cmd[2].q,2)
        b3 = round(self.low_cmd.motor_cmd[3].q,2)
        b4 = round(self.low_cmd.motor_cmd[4].q,2)
        b5 = round(self.low_cmd.motor_cmd[5].q,2)
        b6 = round(self.low_cmd.motor_cmd[6].q,2)
        b7 = round(self.low_cmd.motor_cmd[7].q,2)
        b8 = round(self.low_cmd.motor_cmd[8].q,2)
        b9 = round(self.low_cmd.motor_cmd[9].q,2)
        b10 = round(self.low_cmd.motor_cmd[10].q,2)
        b11 = round(self.low_cmd.motor_cmd[11].q,2)
        #print("cmd",self.low_cmd.motor_cmd[4].q,"state",self.low_state.motor_state[4].q)



        default = [0.1,  0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1, -1.5, -0.1, 1, -1.5]
        for i in range(12):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = default[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = 50
            self.low_cmd.motor_cmd[motor_idx].kd = 5
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        
        #self.send_cmd(self.low_cmd)
        q0 = round(self.low_cmd.motor_cmd[0].q,2)
        q1 = round(self.low_cmd.motor_cmd[1].q,2)
        q2 = round(self.low_cmd.motor_cmd[2].q,2)
        q3 = round(self.low_cmd.motor_cmd[3].q,2)
        q4 = round(self.low_cmd.motor_cmd[4].q,2)
        q5 = round(self.low_cmd.motor_cmd[5].q,2)
        q6 = round(self.low_cmd.motor_cmd[6].q,2)
        q7 = round(self.low_cmd.motor_cmd[7].q,2)
        q8 = round(self.low_cmd.motor_cmd[8].q,2)
        q9 = round(self.low_cmd.motor_cmd[9].q,2)
        q10 = round(self.low_cmd.motor_cmd[10].q,2)
        q11 = round(self.low_cmd.motor_cmd[11].q,2)
        t0 = round(self.low_state.motor_state[0].q,2)
        t1 = round(self.low_state.motor_state[1].q,2)
        t2 = round(self.low_state.motor_state[2].q,2)
        #print(f'\r{b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11}', end='', flush=True)
        r0 = round(self.remote_controller.ly,1)
        r1 = round(self.remote_controller.lx * -1,1)
        r2 = round(self.remote_controller.rx * -1,1)
        #print(f'\r{r0,r1,r2}', end='', flush=True)
        time.sleep(0.002)

        #return torque_0,torque_1,torque_2
        default_0 = self.config.default_angles[0] 
        action_0 = self.action[0] 
        action_scaled_0 = self.action[0] * self.config.action_scale
        target_0 = target_dof_pos[0]
        q_0 = self.low_state.motor_state[3].q
        dq_0 = self.low_state.motor_state[3].dq
        torque_0 = (target_dof_pos[0]-self.low_state.motor_state[3].q)*20 - self.low_state.motor_state[3].dq*2
        return self.obs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="go2.yaml")
    args = parser.parse_args()
    
    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)
    print("LA CONFIGURATION A BIEN ETE CHARGEE")

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)
    print("LE CHANNELFACTORY A ETE CREE")

    controller = Controller(config)
    controller.Init()

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()
    print("LE ZERO TORQUE STATE EST EFFECTUE")

    # Move to the default position
    controller.move_to_default_pos()

    L1 = []
    L2 = []
    L3 = []
    L4 = []
    L5 = []
    L6 = []
    L7 = []
    time_ms = 0  # Temps initial
    Liste_t = []
    while True:

        try:
            obs = controller.run()
            L1.append(obs[0])
            L2.append(obs[1])
            L3.append(obs[2])
            Liste_t.append(time_ms)
            time_ms += 2  # Incrément de 2 ms
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                # Move to a laid position
                controller.move_to_ground()
                break

        except KeyboardInterrupt:
            break

    plt.plot(Liste_t,L1, label="cx")
    plt.plot(Liste_t,L2, label="cy")
    plt.plot(Liste_t,L3, label="cwz")
    plt.legend()
    plt.show()
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
    
