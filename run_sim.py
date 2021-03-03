from IPython.display import clear_output
from kendama_env import KendamaEnv
import numpy as np 
import time


def run():
    env = KendamaEnv()
    a = 0
    start_time = time.time()
    while 1:
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time > 1.0/env.dt:
            a += 0.005
            start_time = time.time()


            # action = [list(np.random.normal(scale = 0,loc=1,size = 3)),list(np.random.normal(scale = 10,size = 3))]
            action = np.array([0,0,0,0,0,0])
            ob, reward, done, _ = env.step(action)
            #print(reward)
            clear_output(wait=True)

            if(done):
                env.close()
                break

if __name__ == "__main__":
    run()

