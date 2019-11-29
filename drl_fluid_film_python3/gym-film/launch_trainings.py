## I wanted this to work like this :
##
## - in the parameter dictionary `parameters`, you can put whatever parameter you want to change
## between trainings, with a list of possible values
## 
## - if you want some parameters to be dependant of one another (numbert of jets and their positions for example)
## then you can, instead of a possible value, write a list of [possible_value, parameters_dictionary] where the 
## parameters dictionary follows the same syntax as `parameters`
##
## - Three (you can launch more) trainings are launched for every combinations of parameters possible given your dictionary
##
## This does not output the best combination of parameters

import os
import subprocess
import fileinput
import time
import threading
i = 2

# parameters = {
#     'method': [['1env_1jet', {'policy_name': ['mlp']}],
#                ['1env_njet', {'policy_name': ['mlp', 'cnn']}]],
#     'simulation_time': [3, 6, 12, 20],
#     'n_jets':[[1, {'position_first_jet':[150, 170, 200]}],
#               [2, {'position_first_jet':[170]}],
#               [5, {'position_first_jet':[170]}],
#               [10, {'position_first_jet':[150]}],
#               [20, {'position_first_jet':[50]}]],
#     'JET_WIDTH_mm':[1, 2, 4, 5],
#     'JET_MAX_POWER': [0.1, 1, 5, 10, 30],
#     'size_obs': [10, 20, 40],
#     'size_obs_to_reward': [10, 20, 40]
# }

# parameters and their values' ranges
parameters = {
    'method': [['1env_1jet', {'policy_name': ['mlp']}],
               ['1env_njet', {'policy_name': ['mlp', 'cnn']}]],
    # 'simulation_time': [3, 6, 12, 20],
    # 'n_jets': [[1, {'position_first_jet': [150, 170, 200]}],
    #            [2, {'position_first_jet': [170]}],
    #            [5, {'position_first_jet': [170]}],
    #            [10, {'position_first_jet': [150]}],
    #            [20, {'position_first_jet': [50]}]],
    # 'JET_WIDTH_mm': [1, 2, 4, 5],
    # 'JET_MAX_POWER': [0.1, 1, 5, 10, 30],
    'size_obs': [10, 20, 40],
    'size_obs_to_reward': [10, 20, 40]
}

# This will be the dictionary that will be changed through the script and param.py will change according to it
current_parameters = {
    'method': "1env_njet",
    'policy_name': "mlp",
    'position_first_jet': 150,
    'simulation_time': 12,
    'n_jets': 5,
    'JET_WIDTH_mm': 5.0,
    'space_between_jets': 10.0,
    'JET_MAX_POWER': 4.0,
    'size_obs': 10.0,
    'size_obs_to_reward': 10.0
}

#subprocess.run(["python3", "start_server.py", "-p", port, "--render", str(args.render)])
#param_file = shutil.copy(models_dir+"param.py", "./")

param_file = "param.py"


def write_to_param_file(key, value):
    for line in fileinput.input(param_file, inplace=True):
        line = line.strip('\n')
        if line.find(key+'=') == 0:
            if isinstance(value, str):
                value = "'"+value+"'"
            print(key+'='+str(value))
        else:
            print(line)
    global current_parameters
    current_parameters[key] = value


def start_training(port, training_name):
    subprocess.run(["nohup", "python3", "train.py", '-t',
                    "-tn", training_name, "-p", str(port)])


def launch_training(port, training_number):
    training_name = str(i) + '-'
    for key, value in current_parameters.items():
        training_name = training_name + '_' + key + '-' + str(value)
    training_name = training_name + '_' + str(training_number)

    threading.Thread(target=start_training,
                     args=(port, training_name)).start()


def launch_trainings(params):
    port = i*10000
    for key, values in params.items():
        for value in values:
            if isinstance(value, list):
                write_to_param_file(key, value[0])
                launch_trainings(value[1])
            else:
                write_to_param_file(key, value)
                launch_training(port, 1)
                port += 1
                time.sleep(10)
                launch_training(port, 2)
                port += 1
                time.sleep(10)
                launch_training(port, 3)
                port += 1
                time.sleep(10)


if __name__ == '__main__':
    launch_trainings(parameters)
