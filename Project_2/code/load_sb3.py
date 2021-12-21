import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
if platform =="darwin": # mac
  #import PyQt5
  #matplotlib.use("Qt5Agg")
  pass
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
# path to saved models, i.e. interm_dir + '120821171005'
log_dir = interm_dir + '120821171005'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {"motor_control_mode":"CARTESIAN_PD",
               "task_env": "LR_COURSE_TASK",
               "observation_space_mode": "LR_COURSE_OBS"}
# env_config = {"motor_control_mode":"CARTESIAN_PD", # TODO
#                "task_env": "LR_COURSE_TASK"}
# env_config = {}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = True
# env_config['add_noise'] = True
env_config['test_env'] = False

# get latest model and normalization stats, and plot
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
<<<<<<< HEAD
# model_name = "./logs/intermediate_models/121121143740/rl_model_810000_steps.zip" # SAC model
=======
model_name = "./logs/intermediate_models/121121143740/rl_model_810000_steps.zip" # SAC model
>>>>>>> f975edaad8c2cfa62f6ed5b2fa9875f015e48bc8
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

# [TODO  initialize arrays to save data from simulation
TEST_STEPS = 1000
joint_pos = np.zeros((TEST_STEPS, 3))
base_pos = np.zeros((TEST_STEPS, 3))
base_vel = np.zeros((TEST_STEPS, 3))
velocities = np.zeros((TEST_STEPS-1, 2, 4))
CoT = np.zeros((TEST_STEPS, 1))
legs_z = np.zeros((TEST_STEPS, 4))
ground_contact = np.zeros((TEST_STEPS, 4))

for i in range(TEST_STEPS):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO : test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    base_pos[i, :] = info[0]["base_pos"]
    base_vel[i, :] = info[0]["base_vel"]
    joint_pos[i, :] = info[0]["joint_pos"]
    CoT[i, :] = info[0]["CoT"]
    legs_z[i, :] = info[0]["legs_z"]
    ground_contact[i, :] = info[0]["ground_contact"]

    if dones:
        print('episode_reward', episode_reward)
        episode_reward = 0
if not dones :
    print('episode_reward', episode_reward)
    episode_reward = 0

    # [TODO  save data from current robot states for plots


mass_env = QuadrupedGymEnv()
print("CoT: ", np.sum(CoT) / ( 9.8 * sum(mass_env.robot.GetTotalMassFromURDF()) *           \
                               base_pos[-1, 0]))
print("Average velocity: ", base_pos[-1, 0]/(TEST_STEPS * 0.001 * 10))
# [TODO  make plots:
fig, ax = plt.subplots(1, 1)

ax.plot(base_vel)
ax.set_xlabel('Steps')
ax.set_ylabel('Linear base velocity')
ax.legend(['x vel', 'y vel', 'z vel'])

fig, ax = plt.subplots(4, 1)
ax[0].plot(ground_contact[:, 0])
ax[0].set_ylabel('FR leg')
ax[1].plot(ground_contact[:, 1])
ax[1].set_ylabel('FL leg')
ax[2].plot(ground_contact[:, 2])
ax[2].set_ylabel('RR leg')
ax[3].plot(ground_contact[:, 3])
ax[3].set_ylabel('RL leg')
ax[3].set_xlabel('steps')

if TEST_STEPS > 250:
	ground_contact = ground_contact[250:TEST_STEPS]
duty_cycles = np.sum(ground_contact, axis=0)/(np.ones((1, 4))* TEST_STEPS)
print("Duty cycles: ", duty_cycles)


plt.show()
<<<<<<< HEAD
=======


>>>>>>> f975edaad8c2cfa62f6ed5b2fa9875f015e48bc8
