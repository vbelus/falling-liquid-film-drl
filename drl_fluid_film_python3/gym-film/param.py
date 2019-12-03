import numpy as np


########################## method used - "1env_1jet", or "1env_njet"  ##################################################
method='1env_1jet'

########################## nature of neural network - "mlp", "mlp_shared" or "cnn"  ##################################################
policy_name='mlp'

########################## architecture of NN - default = [128, 64, 64] ##################################################
## this should be like this: [nb of neurons in first layer, nb of neurons in second layer, ...]
arch=[128, 64, 64]




############################################### about the simulation ##############################################################
###### if you change these params, next time you launch a training, set the following param to 1 ############
new_params = 0  ### will recalculate the initial system_state with the new params #TODO - make that automatic

L = 300.0  # length in mm - default: 300

C = 1e-4 # default: 1e⁻4
dx = 10e-2 # default: 1e⁻1
dt = C / dx # C / dx - default value: 1e-3

NUM = int(L/dx) # number of points in numerical resolution

simulation_step_time = 5.0e-2 # time in s of a step of the simulation - default: 5e-2
n_step = int(simulation_step_time / dt) # number of times the numerical scheme will be applied in a single step
                                        # it will affect how much time passes between each step of the environment (= dt*n_step)
simulation_time=20 # time in s of an episode

initial_waiting_time = 200 # the number of s before saving the system state that will serve as the initial system state for all trainings to come
initial_waiting_n_step = int(initial_waiting_time/simulation_step_time)
########################################################################################################################################



############################################### about the environment ##############################################################
nb_timestep_per_simulation = int(simulation_time/simulation_step_time) # number of steps per epoch

# nb_total_epoch = 10000 # total number of episodes
# save_every_n_epoch = 100 # the model is saved every n episodes

total_nb_timesteps = int(3e5)
nb_saves_per_training = 10
nb_epoch = total_nb_timesteps // nb_timestep_per_simulation
save_every_n_epoch = (nb_epoch-1) // nb_saves_per_training

n_cpu=1
threshold_hq = 5 # max value in obs, to not give too high inputs to the nn

#################### about the jets ##################
n_jets=10
JET_MAX_POWER=5
JET_WIDTH_mm=5.0

space_between_jets=10


position_first_jet=150
JET_WIDTH = int(JET_WIDTH_mm/dx)
jets_position = np.array(
    [position_first_jet + space_between_jets*i for i in range(n_jets)])  # in mm
jets_position = np.array(jets_position/dx, dtype="int")

# we can add perturbation jets to challenge a policy that would adapted only to the case of a an unpertubated simulation
perturbation_jets_position=[]
perturbation_jets_power = JET_MAX_POWER

#################### about the obs/reward ###################
cut_obs_before_jet = 1.0 # to change the size of the jet without changing the position of its left extremity - default: 1

size_obs=20
size_obs_to_reward=20

reward_param = 5.66  # chosen so that a no jet policy gives a reward of ~0
obs_param = 1
nan_punition=-500.0 # reward given when the simulation collapses
true_reset_every_n_episodes = False
############################################################################################################################



########################################## about rendering ##################################################################
render = False

MAX_TIMEFRAME_CONTROL_PLOT = 64  # Max number of points to plot the control+h/time
MAX_TIMEFRAME_FULL_CONTROL_PLOT = 48
POSITION_JET_PLOT = 0.5  # Where to plot the jets and sensors
POSITION_REWARD_SPOTS = 0.4
POSITION_CONTROL_SPOTS = 0.6
N_PLOTS = 3  # nb of plots
show_control = False
# It's where we start the plotting (we're most interested in the zone where waves are already fully formed)
start_h_plot = 0

RENDER_PERIOD=1
SAVE_PERIOD = 1000
obs_at_jet_render_param = 4.0
reward_multiplier_render = 1
########################################################################################################################################




##########################################  parameters that should not be changed ###################################################

obs_h, obs_q = True, True

delta = 0.1
noise_mag = 1e-4

hq_base_value = 1.0
max_h = 1  # Important for plot and obs space
max_q=3

normalize_value_q=1
normalize_value_h=1
########################################################################################################################################


# misc
tensorboard_integration = True
monitor_reward = True
is_dummy_vec_env = False