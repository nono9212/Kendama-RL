import gym
from IPython.display import clear_output
from kendama_env import KendamaEnv
import numpy as np 
import time
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2,SAC
from stable_baselines.common.vec_env import VecNormalize
from IPython.display import clear_output
import os
def make_env( rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = KendamaEnv(render=False)
        return env
    
    return _init

if __name__ == '__main__':
    num_cpu = 8
    env = SubprocVecEnv([make_env( i) for i in range(num_cpu)])
    env = VecNormalize(env, norm_reward= False)
    
    
    #env = VecNormalize.load("vec_normalize.pkl", env)
    #env.norm_reward = False

    model = SAC(MlpPolicySAC, env, verbose=0,tensorboard_log="./log_model/", gamma=0.985, learning_rate=0.001)
    #model = PPO2.load(".ppo2.zip",env, verbose=0,tensorboard_log="./log_model2/", gamma=0.985)


    #model.learn(total_timesteps=10000000)

    model.learn(total_timesteps=20000000,tb_log_name="agent2", reset_num_timesteps=False)
    
    log_dir = "."
    model.save( "sac")
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env.save(stats_path)