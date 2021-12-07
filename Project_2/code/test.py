import time
import numpy as np
import matplotlib
from sys import platform
if platform =="darwin": # mac
  import PyQt5
  matplotlib.use("Qt5Agg")
else: # linux
  matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.quadruped_gym_env import QuadrupedGymEnv
from hopf_network import HopfNetwork



ADD_CARTESIAN_PD = True
SIM_TIME = 10 # [{value for value in variable}]
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

for w_stance in [x/2 for x in range(11,70)]:
    for w_swing in [x/2 for x in range(10,70)]:


        env = QuadrupedGymEnv(
                                # render=True,              # visualize
                                render=False,              # visualize
                                # on_rack=True,              # useful for debugging!
                                on_rack=False,              # not useful for debugging!
                                isRLGymInterface=False,     # not using RL
                                time_step=TIME_STEP,
                                action_repeat=1,
                                motor_control_mode="TORQUE",
                                add_noise=False,    # start in ideal conditions
                                # record_video=True
                                )
        # print("env setup done")

        # initialize Hopf Network, supply gait
        cpg = HopfNetwork(time_step=TIME_STEP,
                            # omega_swing  = 10*np.pi,  # NOTE: modified
                            # omega_stance =  5*np.pi,  # NOTE: modified
                            # gait="TROT",              # change depending on desired gait
                            omega_swing  = w_swing*np.pi,  # NOTE: modified (works okay: 10, 50, 0.07)
                            omega_stance = 40*np.pi,  # NOTE: modified
                            gait="BOUND",             # change depending on desired gait
                            #des_step_len = .06 #
                            # omega_swing  = 15.0*np.pi,  # NOTE: modified
                            # omega_stance =  5.0*np.pi,  # NOTE: modified
                            # gait="WALK",            # change depending on desired gait
                            # omega_swing  = 15.0*np.pi,  # NOTE: modified
                            # omega_stance =  3.0*np.pi,  # NOTE: modified
                            # gait="PACE",            # change depending on desired gait
                            )
        # print("cpg setup done")

        TEST_STEPS = int(SIM_TIME / (TIME_STEP))
        t = np.arange(TEST_STEPS)*TIME_STEP

        # TODO  initialize data structures to save CPG and robot states
        joint_pos = np.zeros((12, TEST_STEPS))
        cpg_states = np.zeros((TEST_STEPS, 2, 4))
        cpg_velocities = np.zeros((TEST_STEPS-1, 2, 4))

        ############## Sample Gains
        # joint PD gains
        kp=np.array([150,70,70])
        kd=np.array([2,0.5,0.5])
        # Cartesian PD gains
        kpCartesian = np.diag([2500]*3)
        kdCartesian = np.diag([40]*3)

        # print("starting sim for " + str(w_swing))
        for j in range(TEST_STEPS):
            # initialize torque array to send to motors
            action = np.zeros(12)
            # get desired foot positions from CPG
            xs,zs = cpg.update()
            # Note: get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
            q  = env.robot.GetMotorAngles()
            dq = env.robot.GetMotorVelocities()

            # loop through desired foot positions and calculate torques
            for i in range(4):
                # initialize torques for legi
                tau = np.zeros(3)
                # get desired foot i pos (xi, yi, zi) in leg frame
                leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])
                # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
                leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz) # NOTE: modified
                # Add joint PD contribution to tau for leg i (Equation 4)
                tau += kp * (leg_q - q[3*i:3*i+3])  # + kd * (0 - dq[3*i:3*i+3]) # TODO: fix this

                # add Cartesian PD contribution
                if ADD_CARTESIAN_PD:
                    # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
                    J, pos = env.robot.ComputeJacobianAndPosition(i) # NOTE: modified
                    # Get current foot velocity in leg frame (Equation 2)
                    vel = np.matmul(J, dq[3*i:3*i+3])
                    # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
                    tau += np.matmul(J.T, np.matmul(kpCartesian,(leg_xyz-pos)) + np.matmul(kdCartesian, (0-vel))) # TODO: fix this

                # Set tau for legi in action vector
                action[3*i:3*i+3] = tau

            # send torques to robot and simulate TIME_STEP seconds
            env.step(action)

            # TODO  save any CPG or robot states
            joint_pos[:,j] = q
            cpg_states[j] = cpg.X
            if j > 0:
                cpg_velocities[j-1] = cpg_states[j] - cpg_states[j-1]
                cpg_velocities[j-1,1,:] = cpg_velocities[j-1,1,:] % (2*np.pi)
                cpg_velocities[j-1] = cpg_velocities[j-1] / cpg._dt


            if env.is_fallen():
                print("robot has fallen")
                break

            if (j > TEST_STEPS/SIM_TIME) and (env.robot.GetBasePosition()[0] < 0):
                print("robot is going backwards")
                break
            if (j == TEST_STEPS/2) and (env.robot.GetBasePosition()[0] < 2):
                print("robot is slow")
                break



        if not env.is_fallen():
            if env.robot.GetBasePosition()[0] > 0 :
                dist = np.sqrt(np.sum(np.array(env.robot.GetBasePosition())**2))
                if dist < 4 :
                    print("robot is slow")
                else:
                    print("\t!! " + str(w_swing) +"/"+ str(w_stance)+ " seems to work. Distance travelled: {:.1f} m".format(dist))


#####################################################
# PLOTS
#####################################################
# example
# fig = plt.figure()
# plt.plot(t,joint_pos[0+1,:], label='FR thigh')
# plt.plot(t,joint_pos[3+1,:], label='FL thigh')
# plt.plot(t,joint_pos[6+1,:], label='RR thigh')
# plt.plot(t,joint_pos[9+1,:], label='RL thigh')
# plt.xlabel("t [s]")
# plt.legend()

# fig = plt.figure()
# ax1 = plt.subplot(2, 2, 1)
# plt.title("amplitude")
# ax1.plot(t,cpg_states[:, 0, :])
# ax1.legend(['FR', 'FL', 'RR', 'RL'])
#
# ax2 = plt.subplot(2, 2, 2)
# plt.title('theta')
# ax2.plot(t,cpg_states[:, 1, :])
# ax2.legend(['FR', 'FL', 'RR', 'RL'])
#
# ax3 = plt.subplot(2, 2, 3)
# plt.title('r velocity')
# ax3.plot(t[0:-1],cpg_velocities[:, 0, :])
# ax3.legend(['FR', 'FL', 'RR', 'RL'])
#
# ax4 = plt.subplot(2, 2, 4)
# plt.title('theta velocity')
# ax4.plot(t[0:-1],cpg_velocities[:, 1, :])
# ax4.legend(['FR', 'FL', 'RR', 'RL'])

plt.show()
