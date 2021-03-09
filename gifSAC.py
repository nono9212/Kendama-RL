import imageio
import gym
from IPython.display import clear_output
from kendama_env import KendamaEnv
import numpy as np 
import time
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2,SAC
from stable_baselines.common.vec_env import VecNormalize
from IPython.display import clear_output


env = DummyVecEnv([lambda : KendamaEnv(render=False)])
env = VecNormalize.load("vec_normalize.pkl", env)
model = SAC.load("sac.zip", env)
images = []
obs = model.env.reset()
img = model.env.render()
for i in range(350):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _ ,_ = model.env.step(action)
    img = model.env.render()

imageio.mimsave('test.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=240)