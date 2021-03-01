import gym
from gym import spaces
import pybullet as p
import time
import pybullet_data
import numpy as np
import scipy.spatial.transform.Rotation as R

TIME_LIM = 300
X_LIM, Y_LIM, Z_LIM = 10, 10, 10

class KendamaEnv(gym.Env):
  """Custom Environment that follows gym interface"""


  def __init__(self):
    super(KendamaEnv, self).__init__()

    # 2 (force, torque) * 3D
    self.action_space = spaces.Box(low=-1, high=1, shape=(2,3), dtype=np.float32)
    
    # 2 objects * 3 (pos, vit, acc) * 3D
    self.observation_space = spaces.Box(low=-1, high=1, shape=(2,3,3), dtype=np.float32)

    # time during the simulation
    self.time = 0

    self.physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    self.planeId = p.loadURDF("plane.urdf")
    cubeStartPos = [0,0,0]
    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    self.ken = p.loadURDF("./URDF/kendama_hand/kendama_hand.urdf",cubeStartPos, cubeStartOrientation, 
                    # useMaximalCoordinates=1, ## New feature in Pybullet
                    flags=p.URDF_USE_INERTIA_FROM_FILE)
    cubeStartPos = [0,0,0.35]
    cubeStartOrientation = p.getQuaternionFromEuler([0,3.1415/2.0,0])
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
    self.dt = 240 # to compute acceleration

    # constraints 
    self.center = p.loadURDF("./URDF/cube/cube.urdf",[0,0,0.1])
    
    p.createConstraint(self.ken, -1, self.center, -1, p.JOINT_POINT2POINT,jointAxis=[1,0,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
    self.pulling = True
    self.pulling2 = False
    self.link = p.createConstraint(self.center, -1, self.dama, -1, p.JOINT_POINT2POINT,jointAxis=[1,1,1],parentFramePosition=[0,0,0],childFramePosition=[0.5,0,0])
    p.changeConstraint(self.link,maxForce=10000)

  # ckecks time limit
  def out_of_time(self):
    return self.time > TIME_LIM
  
  # check spatial limits
  def out_of_box(self, observation):
    kx, ky, kz = np.absolute(observation[0,0])
    dx, dy, dz = np.absolute(observation[1,0])

    if ( kx > X_LIM or dx > X_LIM):
      return False
    if ( ky > Y_LIM or dy > Y_LIM):
      return False
    if ( kz > Z_LIM or dz > Z_LIM):
      return False

    return True


  def step(self, action):
    self.time += 1

    # tension 
    if(self.pulling):
        force = p.getConstraintState(self.link)
    else :
        force = [0,0,0]
    #print("({:.2f} {:.2f} {:.2f})".format(force[0], force[1], force[2]))
    a = np.array(p.getBasePositionAndOrientation(self.center)[0])
    a -= np.array(p.getBasePositionAndOrientation(self.dama)[0])
    # compute the work of the force
    tension = np.dot(a,np.array(force))

    # update the parameters : wire not under tension anymore
    if(tension < 0 and self.pulling):
        self.pulling = False
        p.removeConstraint(self.link)
    # wire back under tension
    if(np.linalg.norm(a) > 0.5 and not self.pulling):
        self.pulling = True
        self.pulling2 = True
        axis = np.dot(np.linalg.inv(np.reshape(p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(self.dama)[1]), [3,3])), a/np.linalg.norm(a))
        self.link = p.createConstraint(self.center, -1, self.dama, -1, p.JOINT_POINT2POINT,axis/2.0, [0, 0, 0],axis/2.0)
        p.changeConstraint(self.link,maxForce=10000)



    kenPos,_ = p.getBasePositionAndOrientation(self.ken)
    p.applyExternalForce(objectUniqueId=self.ken, linkIndex=-1,
                         forceObj=action[0], posObj=kenPos, flags=p.WORLD_FRAME)
    p.applyExternalTorque(objectUniqueId=self.ken, linkIndex=-1,
                         torqueObj=action[1], flags=p.WORLD_FRAME)
    p.stepSimulation()

    # get the parameters after the simulation
    kenPos, kenOr = p.getBasePositionAndOrientation(self.ken)
    kenOr = p.getEulerFromQuaternion(kenOr)
    kenV, kenW = p.getBaseVelocity(self.ken)
    kenAcc = tuple(np.subtract(kenVel, self.v_ken) / (self.dt*1000))
    kenAccRad = tuple(np.subtract(kenVelRad, self.vRad_ken) / (self.dt*1000))
    self.v_ken = kenVel
    self.vRad_ken = kenAccRad

    damaPos, damaOr = p.getBasePositionAndOrientation(self.dama)
    damaOr = p.getEulerFromQuaternion(damaOr)
    damaV, damaW = p.getBaseVelocity(self.dama)
    damaAcc = tuple(np.subtract(damaVel, self.v_dama) / (self.dt*1000))
    damaAccRad = tuple(np.subtract(damaVelRad, self.vRad_dama) / (self.dt*1000))
    self.v_dama = damaVel
    self.vRad_dama = damaAccRad
    
    observation = np.array([
      [kenPos, kenOr, kenV, kenW, kenAcc, kenAccRad],
      [damaPos, damaOr, damaV, damaW, damaAcc, damaAccRad]
      ])

    done = False
    info = False

    # get 
    localOrientation = [0,0,1]
    r = R.from_quat(p.getBasePositionAndOrientation[1])
    localOrientation = r.apply(localOrientation)

    # add final reward
    if self.pulling:
        reward = 
          damaV[2] +
          np.dot(damaV[2],localOrientation[2])**2 -
          lambda_ * (kenPos[2]**2 + np.dot(kenVel, kenVel))
    else:
      reward = 
        (kenPos[0] - damaPos[0])**2 +
        (damaV[:2] - kenOr[:2])**2 + (kenPos[1] - damaPos[1])**2 +
        (kenOr[0] - damaOr[0])**2 + (kenOr[1] - damaOr[1])**2

    # dama is back beneath the ken, then it is a final state
    if self.pulling2 and damaPos[2] < kenpos[2]:
      done = True
      final_reward = reward_list[-1]

    if good_collision # Condition sur la distance entre le centre du ken et le centre du dama : a determiner
      reward = 100
  # Il faut normaliser le return a chaque fois ... : se fait avec les fonctions exponentielles

    if out_of_box():
      done = True
      reward = 0

    if self.out_of_time():
      done = True
      reward = 0
    
    return observation, reward, done, info

  def reset(self):
    self.time = 0
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