import gym
from gym import spaces
import pybullet as p
import time
import pybullet_data
import numpy as np

class KendamaEnv(gym.Env):
  """Custom Environment that follows gym interface"""


  def __init__(self):
    super(KendamaEnv, self).__init__()

    # 2 (force, torque) * 3D
    self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(2,3), dtype=np.float32)
    
    # 2 objects * 3 (pos, vit, acc) * 3D
    self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(2,3,3), dtype=np.float32)

    self.physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    self.planeId = p.loadURDF("plane.urdf")
    cubeStartPos = [0,0,0]
    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    self.ken = p.loadURDF("./URDF/kendama_hand/kendama_hand.urdf",cubeStartPos, cubeStartOrientation, 
                    # useMaximalCoordinates=1, ## New feature in Pybullet
                    flags=p.URDF_USE_INERTIA_FROM_FILE)
    cubeStartPos = [0,0,0.5]
    cubeStartOrientation = p.getQuaternionFromEuler([3.14,0,0])
    self.dama = p.loadURDF("./URDF/kendama_ball/kendama_ball.urdf",cubeStartPos, cubeStartOrientation, 
                    # useMaximalCoordinates=1, ## New feature in Pybullet
                    flags=p.URDF_USE_INERTIA_FROM_FILE)
    p.resetDebugVisualizerCamera(1.06,41,-28.8,[0,0,0])
    self.v_ken = (0,0,0) # to compute acceleration
    self.v_dama = (0,0,0) # to compute acceleration
    self.vRad_dama = (0,0,0) # to compute acceleration
    self.vRad_ken = (0,0,0) # to compute acceleration
    self.dt = 10 # to compute acceleration

  def step(self, action):
    pos_ken,_ = p.getBasePositionAndOrientation(self.ken)
    p.applyExternalForce(objectUniqueId=self.ken, linkIndex=-1,
                         forceObj=action[0], posObj=pos_ken, flags=p.WORLD_FRAME)
    p.applyExternalTorque(objectUniqueId=self.ken, linkIndex=-1,
                         torqueObj=action[1], flags=p.WORLD_FRAME)
    p.stepSimulation()
    kenVel,kenVelRad = p.getBaseVelocity(self.ken)
    kenAcc = tuple(np.subtract(kenVel, self.v_ken) / (self.dt*1000))
    kenAccRad = tuple(np.subtract(kenVelRad, self.vRad_ken) / (self.dt*1000))
    self.v_ken = kenVel
    self.vRad_ken = kenAccRad
    kenPos,kenAngle = p.getBasePositionAndOrientation(self.ken)
    damaVel,damaVelRad = p.getBaseVelocity(self.dama)
    damaAcc = tuple(np.subtract(damaVel, self.v_dama) / (self.dt*1000))
    damaAccRad = tuple(np.subtract(damaVelRad, self.vRad_dama) / (self.dt*1000))
    self.v_dama = damaVel
    self.vRad_dama = damaAccRad
    damaPos,damaAngle = p.getBasePositionAndOrientation(self.dama)
    observation = np.array([[kenPos,kenVel,kenAcc,kenAngle,kenVelRad,kenAccRad],[damaPos, damaVel, damaAcc,damaAngle,damaVelRad,damaAccRad]])
    reward = 1
    done = False
    info = False
    return observation, reward, done, info

  def reset(self):
    kenVel = [0,0,0]
    kenAcc = [0,0,0]
    kenPos = [0,0,0]
    damaVel = [0,0,0]
    damaAcc = [0,0,0]
    damaPos = [0,0,0]
    observation = np.array([[kenPos,kenVel,kenAcc],[damaPos, damaVel, damaAcc]])
    self.v_dama = (0,0,0)
    self.v_ken = (0,0,0)
    self.vRad_dama = (0,0,0)
    self.vRad_ken = (0,0,0)
    return observation  # reward, done, info can't be included
  def close (self):
    p.disconnect()