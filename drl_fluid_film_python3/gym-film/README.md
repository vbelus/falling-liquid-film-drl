### Prerequisites
In `drl_fluid_film_python3/gym-film/gym_film/envs/simulation_solver`, execute the makefile

In this directory, build the project with `pip install -e ./`

### How to have some training
The basic command is `python train.py` but you will need to add some flags :
- `--train` to train your model
- `--render` if you want to render the simulation
- `--training_name [the name of your training]` you will need to specify that so that you can load your models later
- `--load` to load a trained model
- `--epoch [number of epochs]` to specify the number of epochs of the model you want to load
- `--ls` this will list the models saved under the training name you specified

If you want to train a model with the default parameters and render it, you can do it with `python train.py --train --render --training_name my-first-training`

### Change the parameters of the training
All the parameters relevant to the training and simulation are in `param.py`. You will have to change the parameters in this file.
Some important parameters here are :

- The number of cpus on your machine `n_cpu`

-> Parallelizing the training can be useful

- The size of our simulation `L`

-> The default value is `300.0`. `L` is not to be confused with `NUM`, which is the size of the array. `L = NUM * dx`

- The number and position of the jets, `n_jets` and `jets_position`

-> The default value is `np.array([250.0, 254.0])`, which means there are two jets at `x = 250.0` and `x = 254.0`. They are next to each other as `JET_WIDTH = 4`

- The size of the observation (input of the agent), and the size of the observation on which the reward is calculated

-> As it is done now (badly) the observation input is defined as an array, whose size in the simulation is `size_obs = 25.0`, the right end of the array being on the right end of the first jet. With only one jet this is valid, as the jet doesn't need to see what is on his right because he can't control it. With multiple jets this doesn't hold anymore, which is why `cut_obs_before_jet` can help - it changes the position of the right end of the array

As for the observation on which the reward is calculated, it starts from the first jet, with size `size_obs_to_reward`

- The number of points in our observation

-> You can change it with `n_obs_points`. If you want as many points as possible given the length of your simulation's array, just put a very big number and it will get clipped

- The duration of one simulation `simulation_time`

-> The default value is `200.0`, which is twice the time the simulation needs to stabilize (at first there are no waves)

- The duration of the simulation before we begin any training `initial_waiting_time`

-> This must be close to `100` if you want to train your jets directly on a situation where waves are fully formed. This initial waiting time is only processed once, when initializing the environment, so don't worry about any performance loss

- The width and power of the jet `JET_WIDTH` and `JET_MAX_POWER`

-> As it is done right now, the action of the agent is always between `-1` and `1` and is later multiplied by `JET_MAX_POWER` to be applied in the simulation. Right now this value and the width value iare letting the jets modify q with the same amplitude as the waves naturally forming, so it should be enough for control. 

`simulation_step_time` is the duration of one step of the environment. In one such step, we do `n_step` of the numerical scheme

`nb_total_epoch` and `save_every_n_epoch` are relevant to training, an epoch being a full simulation 

### On UiO's PCs

- `module load python/anaconda3`
- `source activate py36`              #py36 is the name of my anaconda env, make one with python3 so you can use pip
