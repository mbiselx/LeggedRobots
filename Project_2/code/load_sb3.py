import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
if platform =="darwin": # mac
  import PyQt5
  matplotlib.use("Qt5Agg")
else: # linux
  matplotlib.use('TkAgg')

# stable baselines
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.cmd_util import make_vec_env

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "PPO"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '111121133812'
# log_dir = interm_dir + 'test2'
log_dir = interm_dir + '120721155156'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {"motor_control_mode"      : "CARTESIAN_PD",
               "task_env"               : "LR_COURSE_TASK",
               "observation_space_mode" : "LR_COURSE_OBS"}
# env_config = {"motor_control_mode":"CARTESIAN_PD", # TODO
#                "task_env": "LR_COURSE_TASK"}
# env_config = {}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False
# env_config['add_noise'] = True
# env_config['test_env'] = True

# get latest model and normalization stats, and plot
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show()

# reconstruct env
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# NOTE: initialize arrays to save data from simulation
TEST_STEPS = 2000
joint_pos = np.zeros((12, TEST_STEPS))
states = np.zeros((TEST_STEPS, 2, 4))
velocities = np.zeros((TEST_STEPS-1, 2, 4))
energy = 0

for i in range(TEST_STEPS):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO : test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    # print("info " + str(info))
    if dones:
        print('episode_reward', episode_reward)
        episode_reward = 0

if not dones :
    print('episode_reward', episode_reward)
    episode_reward = 0








#####################################################
# PLOTS
#####################################################
