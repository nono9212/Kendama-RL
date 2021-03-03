import gym
from IPython.display import clear_output
from kendama_env import KendamaEnv
import numpy as np 
import time
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


if __name__ == '__main__': 
    env = KendamaEnv()
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=100000)

