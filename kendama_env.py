import gym
from gym import spaces
import pybullet as p
import time
import pybullet_data
import numpy as np
from IPython.display import clear_output
TIME_LIM = 240*3
X_LIM, Y_LIM, Z_LIM = 1, 1, 3
INITIAL_KEN_POS, INITIAL_KEN_OR = [0,0,0.7], [0,0,0]
INITIAL_DAMA_POS, INITIAL_DAMA_OR = [0,0,0.3], [0, 3.14/2, 0]
class KendamaEnv(gym.Env):
  """Custom Environment that follows gym interface"""

  def __init__(self, render=True):
    super(KendamaEnv, self).__init__()
    self.dt = 240.0
    # 2 (force, torque) * 3D
    self.action_space = spaces.Box(low=np.array([-1,-1,-1,-1,-1,-1]), high=np.array([1,1,1,1,1,1]))
    # 2 objects * 3 (pos, vit, acc) * 3D
    self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(4,3), dtype=np.float32)

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
    self.wasHigher = False

    self.targetKenPos = np.array(INITIAL_KEN_POS)
    self.targetKenAng = np.array(INITIAL_KEN_OR)
    # Fonctionnement du pulling
    if(self.pulling):
      posAttacheDama, angleDama = p.getBasePositionAndOrientation(self.dama)
      posAttacheDama = np.array(posAttacheDama) + np.matmul(np.linalg.inv(np.reshape(p.getMatrixFromQuaternion(angleDama),[3,3])), np.array([-0.03,0,0]))
      posCenterCube, angleCenterCube = p.getBasePositionAndOrientation(self.center)
      posCenterCube = np.array(posCenterCube)
      vec = posAttacheDama - posCenterCube
      vec = np.matmul(vec, np.reshape(p.getMatrixFromQuaternion(angleCenterCube),[3,3])) # vec dans le référentiel du cube

      self.link = p.createConstraint(self.center, -1, self.dama, -1, p.JOINT_POINT2POINT,[1,1,1], vec/np.linalg.norm(vec)*0.5, [-0.03,0,0])
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
    #if ( kz > Z_LIM or dz < Z_LIM):
    if(p.getBasePositionAndOrientation(self.ken)[0][2] < 0):
      return True

    return False

  def step(self, action):
    #Resizing action 
    action[0] /= 240.0/24.0
    action[1] /= 240.0/24.0
    action[2] /= 240.0/24.0

    action[3], action[4], action[5] = action[3]*3.14*20.0/240.0, action[4]*3.14/2.0*40.0/240.0, action[5]*3.14*20.0/240.0


    self.time += 1
    # tension 
    if(self.pulling):
        force = p.getConstraintState(self.link)
    else :
        force = [0,0,0]
    
    # We work the dynamics of the system
    posAttacheDama, angleDama = p.getBasePositionAndOrientation(self.dama)
    posAttacheDama = np.array(posAttacheDama)  + np.matmul(np.array([-0.03,0,0]),np.linalg.inv(np.reshape(p.getMatrixFromQuaternion(angleDama),[3,3])))
    posCenterCube, angleCenterCube = p.getBasePositionAndOrientation(self.center)
    posCenterCube = np.array(posCenterCube)

    vec = posAttacheDama - posCenterCube
    vec = np.matmul(vec,np.reshape(p.getMatrixFromQuaternion(angleCenterCube),[3,3])) # vecteur dans le référentiel du cube

    
    dirFil = posAttacheDama - posCenterCube
    tension = np.dot(dirFil/np.linalg.norm(dirFil),np.array(force))
    # What happens when there is no more tension
    if( self.pulling and tension > 0): #np.linalg.norm(vec)<0.5
        self.pulling = False
        p.removeConstraint(self.link)
    
    # What happens when tension comes back
    elif(np.linalg.norm(vec) > 0.5 and not self.pulling):
        self.pulling = True
        self.pulling2 = True
        self.link = p.createConstraint(self.center, -1, self.dama, -1, p.JOINT_POINT2POINT,[1,1,1], vec/np.linalg.norm(vec)*0.5, np.array([-0.03,0,0]))
        p.changeConstraint(self.link,maxForce=100)

    
    # Friction when there is tension
    if self.pulling : 
        friction = 0.1
        proj_vitesse = np.dot(dirFil/np.linalg.norm(dirFil), np.array(p.getBaseVelocity(self.dama)[0])) * dirFil/np.linalg.norm(dirFil)
        friction_force = - friction * (proj_vitesse - np.array(p.getBaseVelocity(self.ken)[0]))
        p.applyExternalForce(objectUniqueId=self.dama, linkIndex=-1,
                         forceObj=friction_force, posObj=np.array(p.getBasePositionAndOrientation(self.dama)[0]), flags=p.WORLD_FRAME)
    
    #>Friction to dissipate energy
    friction = 0.03
    friction_force = - friction * np.array(p.getBaseVelocity(self.dama)[0])
    p.applyExternalForce(objectUniqueId=self.dama, linkIndex=-1,
                      forceObj=friction_force, posObj=np.array(p.getBasePositionAndOrientation(self.dama)[0]), flags=p.WORLD_FRAME) 

    # Action du kendama : Changement de la contrainte
    
    prvPos, prvAngle = self.targetKenPos, self.targetKenAng
    prvPos = prvPos + np.array(action[:3])
    prvAngle = prvAngle + np.array(action[3:])
    n_orn = p.getQuaternionFromEuler(prvAngle)
    self.targetKenPos = prvPos
    self.targetKenAng = prvAngle
    p.changeConstraint(self.ken_constraint, prvPos, jointChildFrameOrientation=n_orn, maxForce=30)

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
    observation = np.array([damaPos - kenPos, damaVel - kenVel, damaAngle - kenAngle, damaVelRad - kenVelRad])
    observation = self.normalizeObs(observation)
    reward, done = self.get_reward(damaPos, kenPos, damaVel, kenVel, damaAngle, kenAngle, damaVelRad, kenVelRad,action)
    
    return observation, reward, done, {}
  
  def renorm_function(self, x, alpha):
    '''
    Renormalizing function between -1  and 1
    '''
    return 2 * ((1 / (1 + np.exp(-alpha*x))) - 0.5)

  def normalizeObs(self, obs):
    obs[0] = self.renorm_function(obs[0], alpha=5)
    obs[1] = self.renorm_function(obs[1], alpha=1)

    obs[2] = np.clip(np.multiply(obs[2], [1.0/3.14, 2.0/3.14, 1.0/3.14]),-1,1)
    obs[3] = self.renorm_function(obs[3], alpha=0.2/3.14)

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
    self.wasHigher = False

    return observation  # reward, done, info can't be included
  
  
  def close (self):
    p.disconnect()


  def iscolliding(self):
    '''
    Determine if there is a collision between the kendama and the ball
    '''
    localOrientation = np.array([0,0,1])
    r = np.reshape(p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(self.ken)[1]),[3,3])
    vect_ken_orientation = np.matmul(localOrientation, np.linalg.inv(r))

    pdama = np.array(p.getBasePositionAndOrientation(self.dama)[0]).copy()
    pken = np.array(p.getBasePositionAndOrientation(self.ken)[0])
    pken += vect_ken_orientation / np.linalg.norm(vect_ken_orientation) * 0.077 # Spike middle position
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
    rwdl = []
    # Je ne sais pas ce que cela fait...
    localOrientation = np.array([0,0,1])
    r = np.reshape(p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(self.dama)[1]),[3,3])
    localOrientation = np.dot(r, localOrientation)

    # Ball is still under control while above dama : this is possible but not desired (Physics are not working perfectly)
    if (damaPos[2] < kenPos[2]+0.03) and self.pulling:
      return 0, True

    # Dama is under the ken
    if damaPos[2] < kenPos[2]:
        reward += damaPos[2]
        if(self.wasHigher):
          return 0, True

    # Dama is above the ken
    else :
      #if(not self.wasHigher):
        #reward += 100.0
      #reward += 1.0
      self.wasHigher = True
      
      
      #Reward sur la différence en position horizontale
      localOrientation = np.array([0,0,1])
      r = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(kenAngle)),[3,3])
      vect_ken_orientation = np.matmul(localOrientation, np.linalg.inv(r))
      spike_ken = vect_ken_orientation / np.linalg.norm(vect_ken_orientation) * 0.077 # Spike middle position
      
      reward += np.exp(-15.0*np.linalg.norm(damaPos - kenPos + spike_ken)**2)
      reward += 10 * np.exp(-500.0*np.linalg.norm(damaPos - kenPos + spike_ken)**2)
      
      #Reward sur la différence en orientation
      localOrientation = np.array([1,0,0])
      r = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(damaAngle)),[3,3])
      vect_dama_orientation = np.matmul(localOrientation, np.linalg.inv(r))
      
      localOrientation = np.array([0,0,1])
      r = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(kenAngle)),[3,3])
      vect_ken_orientation = np.matmul(localOrientation, np.linalg.inv(r))
      
      reward += 2 * np.exp(-10*(np.dot(vect_ken_orientation, vect_dama_orientation)+1)**2)
      #Reward sur l'anticolinéarité entre vecteur vitesse du dama et vecteur orientation du ken
      #rwdl.append(np.exp(-10*(np.dot(vect_ken_orientation, vect_dama_orientation)+1)**2))
      ##reward += np.exp(-10*(np.dot(vect_ken_orientation, damaVel/np.linalg.norm(damaVel))+1)**2) # exp( - (u*v +1)**2 )
      #rwdl.append(np.exp(-10*(np.dot(vect_ken_orientation, damaVel/np.linalg.norm(damaVel))+1)**2))
    #Finally reward for staying around [0,0,1]
    #reward += 0.3*np.exp(- 5.0* np.linalg.norm(kenPos - np.array([0,0,1]))**2)
    #rwdl.append(0.3*np.exp(- 5.0* np.linalg.norm(kenPos - np.array([0,0,1]))**2))

    #Experimental : give reward when no tension in the string
    #if(not self.pulling and self.wasHigher):
    #  reward *= 1.5


    if self.iscolliding(): # Condition sur la distance entre le centre du ken et le centre du dama : a determiner
      reward = 10000.0
      done = True
      print("IT'S A CATCH !")

    if self.out_of_box():
      done = True
      reward = 0

    if self.out_of_time():
      done = True
      reward = 0
    
    
    return reward, done

