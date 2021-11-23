"""
CPG in polar coordinates based on:
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
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


class HopfNetwork():
  """ CPG network based on hopf polar equations mapped to foot positions in Cartesian space.

  Foot Order is FR, FL, RR, RL
  (Front Right, Front Left, Rear Right, Rear Left)
  """
  def __init__(self,
                mu=1**2,                # converge to sqrt(mu)
                omega_swing  = 10*np.pi,  # NOTE: modified
                omega_stance =  5*np.pi,  # NOTE: modified
                gait="TROT",            # change depending on desired gait
                coupling_strength=1,    # coefficient to multiply coupling matrix
                couple=True,            # should couple
                time_step=0.001,        # time step
                ground_clearance=0.05,  # foot swing height
                ground_penetration=0.01,# foot stance penetration into ground
                robot_height=0.25,      # in nominal case (standing)
                des_step_len=0.04,      # desired step length
                ):

    ###############
    # initialize CPG data structures: amplitude is row 0, and phase is row 1
    self.X = np.zeros((2,4))

    # save parameters
    self._mu = mu
    self._omega_swing = omega_swing
    self._omega_stance = omega_stance
    self._couple = couple
    self._coupling_strength = coupling_strength
    self._dt = time_step
    self._set_gait(gait)

    # set oscillator initial conditions
    self.X[0,:] = np.random.rand(4) * .1
    self.X[1,:] = self.PHI[0,:]

    # save body and foot shaping
    self._ground_clearance = ground_clearance
    self._ground_penetration = ground_penetration
    self._robot_height = robot_height
    self._des_step_len = des_step_len

  def _skew(self, FR=0, FL=0, RR=0, RL=0):
    """
    make the skew-symmetric matrix PHI for a gait
    """
    base = np.array([[ FR, FL, RR, RL]])
    a = base - base.T
    return a

  def _set_gait(self,gait):
    """ For coupling oscillators in phase space.
    TODO update all coupling matrices
    Foot Order is FR, FL, RR, RL
    """

    self.PHI_trot  = 2*np.pi * (self._skew( 0.0,  0.5,  0.5,  0.0) + .0)

    self.PHI_bound = 2*np.pi * (self._skew( 0.0,  0.0,  0.5,  0.5) - .0)

    self.PHI_walk  = 2*np.pi * (self._skew( 0.0, -0.5, -.75, -.25) + .0)

    self.PHI_pace  = 2*np.pi * (self._skew( 0.0,  0.5,  0.0,  0.5) + .0)

    if gait == "TROT":
      #print('TROT')
      self.PHI = self.PHI_trot
    elif gait == "PACE":
      #print('PACE')
      self.PHI = self.PHI_pace
    elif gait == "BOUND":
      #print('BOUND')
      self.PHI = self.PHI_bound
    elif gait == "WALK":
      #print('WALK')
      self.PHI = self.PHI_walk
    else:
      raise ValueError( gait + 'not implemented.')
    # print(self.PHI)

  def update(self):
    """ Update oscillator states. """

    # update parameters, integrate
    self._integrate_hopf_equations()

    # map CPG variables to Cartesian foot xz positions (Equations 8, 9)
    x = -self._des_step_len * self.X[0,:]*np.cos(self.X[1,:]) # NOTE: modified
    z = -self._robot_height + (np.sin(self.X[1,:]) > 0).choose([self._ground_penetration, self._ground_clearance])*np.sin(self.X[1,:])   # NOTE: modified

    return x, z


  def _integrate_hopf_equations(self):
    """ Hopf polar equations and integration. Use equations 6 and 7. """
    # bookkeeping - save copies of current CPG states
    X = self.X.copy()
    X_dot = np.zeros((2,4))
    alpha = 50

    # loop through each leg's oscillator
    for i in range(4):
      # get r_i, theta_i from X
      r, theta = X[0,i], X[1,i] # NOTE: modified
      # compute r_dot (Equation 6)
      r_dot = alpha * (self._mu - r**2)*r # NOTE: modified
      # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
      if theta <= np.pi: # NOTE: modified
        theta_dot = self._omega_swing
      else:
        theta_dot = self._omega_stance

      # loop through other oscillators to add coupling (Equation 7)
      if self._couple:
        theta_dot += np.sum(self._coupling_strength * X[0,:] * np.sin(X[1,:] - theta - self.PHI[i,:]) ) # NOTE: modified

      # set X_dot[:,i]
      X_dot[:,i] = [r_dot, theta_dot]

    # integrate
    self.X = self.X + X_dot * self._dt # NOTE: modified
    # mod phase variables to keep between 0 and 2pi
    self.X[1,:] = self.X[1,:] % (2*np.pi)


if __name__ == "__main__":

  ADD_CARTESIAN_PD = True
  SIM_TIME = 10 # [{value for value in variable}]
  TIME_STEP = 0.001
  foot_y = 0.0838 # this is the hip length
  sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

  env = QuadrupedGymEnv(render=True,              # visualize
                      # on_rack=True,              # useful for debugging!
                      on_rack=False,              # not useful for debugging!
                      isRLGymInterface=False,     # not using RL
                      time_step=TIME_STEP,
                      action_repeat=1,
                      motor_control_mode="TORQUE",
                      add_noise=False,    # start in ideal conditions
                      # record_video=True
                      )
  #print("env setup done")

  # initialize Hopf Network, supply gait
  cpg = HopfNetwork(time_step=TIME_STEP,
                    # omega_swing  = 10*np.pi,  # NOTE: modified
                    # omega_stance =  5*np.pi,  # NOTE: modified
                    # gait="TROT",              # change depending on desired gait
                    omega_swing  = 20*np.pi,  # NOTE: modified (works okay: 10, 50, 0.07)
                    omega_stance = 8*np.pi,  # NOTE: modified
                    gait="BOUND",             # change depending on desired gait
                    #des_step_len = .06 #
                    # omega_swing  = 15.0*np.pi,  # NOTE: modified
                    # omega_stance =  5.0*np.pi,  # NOTE: modified
                    # gait="WALK",            # change depending on desired gait
                    # omega_swing  = 15.0*np.pi,  # NOTE: modified
                    # omega_stance =  3.0*np.pi,  # NOTE: modified
                    # gait="PACE",            # change depending on desired gait
                    )
  #print("cpg setup done")

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

  print("starting sim")
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
