import gym
from gym import spaces
import pybullet as p
import time
import pybullet_data
import numpy as np
TIME_LIM = 240*3
X_LIM, Y_LIM, Z_LIM = 1, 1, 3
INITIAL_KEN_POS, INITIAL_KEN_OR = [0,0,1], [0,0,0]
INITIAL_DAMA_POS, INITIAL_DAMA_OR = [0,0,0.6], [0, 3.14/2, 0]
class KendamaEnv(gym.Env):
  """Custom Environment that follows gym interface"""


  def __init__(self, render=True):
    super(KendamaEnv, self).__init__()
    self.dt = 240.0
    # 2 (force, torque) * 3D
    self.action_space = spaces.Box(low=np.array([-1,-1,-1,-1,-1,-1]), high=np.array([1,1,1,1,1,1]))
    # 2 objects * 3 (pos, vit, acc) * 3D
    self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(10,3), dtype=np.float32)

    if(render):
      self.physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    else:
      self.physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)

    # Downloading Ken and setting constraints
    cubeStartPos = INITIAL_KEN_POS
    cubeStartOrientation = p.getQuaternionFromEuler(INITIAL_KEN_OR)
    self.ken = p.loadURDF("./URDF/kendama_hand/kendama_hand.urdf",cubeStartPos, cubeStartOrientation, 
                    flags=p.URDF_USE_INERTIA_FROM_FILE)        
    self.ken_constraint = p.createConstraint(self.ken, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0,0,0], cubeStartPos)

    # Downloading dama
    cubeStartPos = INITIAL_DAMA_POS
    cubeStartOrientation = p.getQuaternionFromEuler(INITIAL_DAMA_OR)
    self.dama = p.loadURDF("./URDF/kendama_ball/kendama_ball.urdf",cubeStartPos, cubeStartOrientation, 
                    flags=p.URDF_USE_INERTIA_FROM_FILE)
    
    self.v_ken = (0,0,0) # to compute acceleration
    self.v_dama = (0,0,0) # to compute acceleration
    self.vRad_dama = (0,0,0) # to compute acceleration
    self.vRad_ken = (0,0,0) # to compute acceleration
    self.dt = 240 # to compute acceleration
    
    # constraints and center
    self.center = p.loadURDF("./URDF/cube/cube.urdf",INITIAL_KEN_POS+np.array([0,0,0.1]),flags=p.URDF_USE_INERTIA_FROM_FILE)
    p.createConstraint(self.ken, -1, self.center, -1, p.JOINT_POINT2POINT,jointAxis=[1,0,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
    self.pulling = False
    self.pulling2 = False

    # Fonctionnement du pulling
    if(self.pulling):
      posAttacheDama, angleDama = p.getBasePositionAndOrientation(self.dama)
      posAttacheDama = np.array(posAttacheDama) + np.dot(np.linalg.inv(np.reshape(p.getMatrixFromQuaternion(angleDama),[3,3])),np.array([0.03,0,0]))
      posCenterCube, angleCenterCube = p.getBasePositionAndOrientation(self.center)
      posCenterCube = np.array(posCenterCube)
      vec = posAttacheDama - posCenterCube
      vec = np.matmul(np.reshape(p.getMatrixFromQuaternion(angleCenterCube),[3,3]),vec)

      self.link = p.createConstraint(self.center, -1, self.dama, -1, p.JOINT_POINT2POINT,[1,1,1], vec/np.linalg.norm(vec)*0.5, [0.03,0,0])
      p.changeConstraint(self.link,maxForce=10)
    
    self.time = 0
    self.list_reward = []
    # Configuration of the visualisation
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # On saute les premières étapes
    #for _ in range(5000):
      #p.stepSimulation()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(1.18,40.2,-25.6,[0,0,0.82])


  def out_of_time(self):
    '''
    Checks if the simulation has finished 
    '''
    return self.time > TIME_LIM
  
  # check spatial limits
  def out_of_box(self):
    '''
    Checks if the ken or the dama are outside from the box
    '''
    kx, ky, kz = np.absolute(p.getBasePositionAndOrientation(self.ken)[0])
    dx, dy, dz = np.absolute(p.getBasePositionAndOrientation(self.dama)[0])

    if ( kx > X_LIM or dx > X_LIM):
      return True
    if ( ky > Y_LIM or dy > Y_LIM):
      return True
    if ( kz > Z_LIM or dz < Z_LIM):
      return False

    return False

  def step(self, action):
    #Resizing action 
    action[0] /= 2.0 #to get -0.5 - 0.5
    action[1] /= 2.0

    action[2] /= 2.0
    action[2] += 1.0

    action[3], action[4], action[5] = action[3]*3.14, action[4]*3.14/2.0, action[5]*3.14


    self.time += 1
    # tension 
    if(self.pulling):
        force = p.getConstraintState(self.link)
    else :
        force = [0,0,0]
    
    # We work the dynamics of the system
    posAttacheDama, angleDama = p.getBasePositionAndOrientation(self.dama)
    posAttacheDama = np.array(posAttacheDama) + np.dot(np.linalg.inv(np.reshape(p.getMatrixFromQuaternion(angleDama),[3,3])),np.array([0.03,0,0]))
    posCenterCube, angleCenterCube = p.getBasePositionAndOrientation(self.ken)
    posCenterCube = np.array(posCenterCube)
    vec = posAttacheDama - posCenterCube
    vec = np.matmul(np.reshape(p.getMatrixFromQuaternion(angleCenterCube),[3,3]),vec)

    dirFil = posAttacheDama - posCenterCube
    tension = np.dot(dirFil/np.linalg.norm(dirFil),np.array(force))

    # What happens when there is no more tension
    if(tension < 0 and self.pulling ):
        self.pulling = False
        p.removeConstraint(self.link)

    # What happens when tension comes back
    if(np.linalg.norm(vec) > 0.5 and not self.pulling):
        self.pulling = True
        self.pulling2 = True
        self.link = p.createConstraint(self.center, -1, self.dama, -1, p.JOINT_POINT2POINT,[1,1,1], vec/np.linalg.norm(vec)*0.5, [0.03,0,0])
        p.changeConstraint(self.link,maxForce=10)
    
    # Friction when there is tension
    if np.linalg.norm(vec) > 0.5 : 
        friction = 0.05
        friction_force = - friction * (np.array(p.getBaseVelocity(self.dama)[0]) - np.array(p.getBaseVelocity(self.ken)[0]))
        p.applyExternalForce(objectUniqueId=self.dama, linkIndex=-1,
                         forceObj=friction_force, posObj=np.array(p.getBasePositionAndOrientation(self.dama)[0]), flags=p.WORLD_FRAME)
    
    # Action du kendama : Changement de la contrainte
    n_orn = p.getQuaternionFromEuler(action[3:])
    p.changeConstraint(self.ken_constraint, action[:3], jointChildFrameOrientation=n_orn, maxForce=10)

    # Make pybullet simulation work
    p.stepSimulation()

    # Get observations from the action result
    # Get ken informations
    kenVel,kenVelRad = p.getBaseVelocity(self.ken)
    kenVel, kenVelRad = np.array(kenVel), np.array(kenVelRad)
    kenAcc = np.subtract(kenVel, self.v_ken) / (self.dt*1000)
    # kenAccRad = np.subtract(kenVelRad, self.vRad_ken) / (self.dt*1000)
    self.v_ken = kenVel
    self.vRad_ken = kenVelRad
    kenPos,kenAngle = p.getBasePositionAndOrientation(self.ken)
    kenPos, kenAngle = np.array(kenPos), np.array(kenAngle)

    # Get dama observations
    damaVel,damaVelRad = p.getBaseVelocity(self.dama)
    damaVel, damaVelRad = np.array(damaVel), np.array(damaVelRad)
    damaAcc = np.subtract(damaVel, self.v_dama) / (self.dt*1000)
    # damaAccRad = np.subtract(damaVelRad, self.vRad_dama) / (self.dt*1000)
    self.v_dama = damaVel
    self.vRad_dama = damaVelRad
    damaPos,damaAngle = p.getBasePositionAndOrientation(self.dama)
    damaAngle = p.getEulerFromQuaternion(damaAngle)
    kenAngle = p.getEulerFromQuaternion(kenAngle)
    kenAngle = np.array(kenAngle)
    damaPos, damaAngle = np.array(damaPos), np.array(damaAngle)
    observation = np.array([kenPos,kenVel,kenAcc,kenAngle,kenVelRad,damaPos,damaVel,damaAcc,damaAngle,damaVelRad])
    observation = self.normalizeObs(observation)
    reward, done = self.get_reward(damaPos, kenPos, damaVel, kenVel, damaAngle, kenAngle, damaVelRad, kenVelRad,action)
    return observation, reward, done, {}
  
  def normalizeObs(self, obs):
    obs[0] = np.clip((obs[0] + np.array([0,0, -1]))*2.0,-1,1)
    obs[5] = np.clip((obs[5] + np.array([0,0, -1]))*2.0,-1,1)

    obs[1], obs[6] = np.clip(obs[1]/2.0,-1,1), np.clip(obs[6]/2.0,-1,1)
    obs[2], obs[7] = np.clip(obs[2]/4.0,-1,1), np.clip(obs[7]/4.0,-1,1)

    obs[3] = np.clip(np.multiply(obs[3], [1.0/3.14, 2.0/3.14, 1.0/3.14]),-1,1)
    obs[8] = np.clip(np.multiply(obs[8], [1.0/3.14, 2.0/3.14, 1.0/3.14]),-1,1)

    obs[4] = np.clip(np.multiply(obs[4], [1.0/3.14/3.0, 2.0/3.14/3.0, 1.0/3.14/3.0]),-1,1)
    obs[9] = np.clip(np.multiply(obs[9], [1.0/3.14/3.0, 2.0/3.14/3.0, 1.0/3.14/3.0]),-1,1)

    return obs

  def reset(self):
    '''
    Reset the whole environment
    '''
    cubeStartPos = INITIAL_KEN_POS
    cubeStartOrientation = p.getQuaternionFromEuler(INITIAL_KEN_OR)
    p.resetBasePositionAndOrientation(self.ken, cubeStartPos, cubeStartOrientation)
    cubeStartPos = INITIAL_DAMA_POS
    cubeStartOrientation = p.getQuaternionFromEuler(INITIAL_DAMA_OR)
    p.resetBasePositionAndOrientation(self.dama, cubeStartPos, cubeStartOrientation)
    p.resetBasePositionAndOrientation(self.center, INITIAL_KEN_POS + np.array([0,0,0.1]), p.getQuaternionFromEuler([0,0,0]))
    self.time = 0
    kenVel,kenVelRad = p.getBaseVelocity(self.ken)
    kenVel, kenVelRad = np.array(kenVel), np.array(kenVelRad)
    self.v_ken = kenVel
    self.vRad_ken = kenVelRad
    kenPos,kenAngle = p.getBasePositionAndOrientation(self.ken)
    kenPos, kenAngle = np.array(kenPos), np.array(kenAngle)
    damaVel,damaVelRad = p.getBaseVelocity(self.dama)
    damaVel, damaVelRad = np.array(damaVel), np.array(damaVelRad)
    self.v_dama = damaVel
    self.vRad_dama = damaVelRad
    damaPos,damaAngle = p.getBasePositionAndOrientation(self.dama)
    damaAngle = p.getEulerFromQuaternion(damaAngle)
    kenAngle = p.getEulerFromQuaternion(kenAngle)
    damaPos, damaAngle = np.array(damaPos), np.array(damaAngle)
    observation = np.array([kenPos,kenVel,[0,0,0],kenAngle,kenVelRad,damaPos,damaVel,[0,0,0],damaAngle,damaVelRad])
    self.v_dama = (0,0,0)
    self.v_ken = (0,0,0)
    self.vRad_dama = (0,0,0)
    self.vRad_ken = (0,0,0)

    return observation  # reward, done, info can't be included
  
  
  def close (self):
    p.disconnect()


  def iscolliding(self):
    '''
    Determine if there is a collision between the kendama and the ball
    '''
    pdama = np.array(p.getBasePositionAndOrientation(self.dama)[0]).copy()
    pdama[2] -= 0.077
    pken = np.array(p.getBasePositionAndOrientation(self.ken)[0])
    return np.linalg.norm(pdama-pken) < 0.005 # Threshold has been determined after some tests

  def get_reward(self, damaPos, kenPos, damaVel, kenVel, damaAngle, kenAngle, damaVelRad, kenVelRad,action):
    '''
    Every args must be an array (except done which is a bool)
    Dimensions of arrays : 
    Pos : 3d ;
    Vel : 3d ;
    Angle : 3d (euler);
    VelRad : 3d (angular vel).
    '''
    done = False
    reward = 0

    # Je ne sais pas ce que cela fait...
    localOrientation = np.array([0,0,1])
    r = np.reshape(p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(self.dama)[1]),[3,3])
    localOrientation = np.dot(r, localOrientation)


    # Dama is under the ken
    if damaPos[2] < kenPos[2]:
        reward = 0

    # Dama is above the ken
    else :
      
      #Reward sur la différence en position horizontale
      #reward += np.exp(-np.linalg.norm(damaPos[0:2] - kenPos[0:2])**2)
      #Reward sur la différence en orientation
      reward += np.exp(-2.0*np.linalg.norm(damaAngle[0:2] - kenAngle[0:2] + np.array([0,3.14/2.0]))**2)

      #Reward sur l'anticolinéarité entre vecteur vitesse du dama et vecteur orientation du ken
      localOrientation = np.array([0,0,-1])
      r = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(kenAngle)),[3,3])
      vect_ken_orientation = np.matmul(r, localOrientation)
      reward += np.exp(-2.0*(np.dot(vect_ken_orientation, damaVel/np.linalg.norm(damaVel))+1)**2) # exp( - (u*v +1)**2 )
    
    #Finally reward for not moving to much and staying around [0,0,1]
    reward += np.exp(- 40.0* np.linalg.norm(kenPos - np.array([0,0,1]))**2)
    reward += np.exp(- 400.0* np.linalg.norm(kenPos - np.array(action[:3]))**2)
    reward += np.exp(- 400.0* np.linalg.norm(kenAngle - np.array(action[3:]))**2)

    if self.iscolliding(): # Condition sur la distance entre le centre du ken et le centre du dama : a determiner
      reward = 100
      done = True
      print("IT'S A CATCH !")

    if self.out_of_box():
      done = True
      reward = 0

    if self.out_of_time():
      done = True
      reward = 0

    return reward, done

