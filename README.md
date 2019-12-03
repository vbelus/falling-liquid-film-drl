# Deep Reinforcement Learning control of the unstable falling liquid film

![no-control](https://media.giphy.com/media/cO3IdQaGRyK0BYAslz/giphy.gif)
![control](https://media.giphy.com/media/dY0qmKEb5bXhf3gHvs/giphy.gif)

I used this code in the paper `Exploiting locality and physical invariants to design effective Deep Reinforcement Learning control of the unstable falling liquid film` that you can check out [**HERE**](XX)

This README will be in two parts :
- The first part will be about the code in itself
- The second part will be about getting an environment ready to run the code (either in a docker container, or in Ubuntu)

## What you can find in this repo

The code is in `drl_fluid_film_python3/gym-film`. You find the following files:
- `param.py` contains the parameters that your next training will use (all the parameters of the project are centralized here)
- `train.py` is the script you will use to train or render a model

In `gym_film`, you will find the following directories:
- `envs`, where our basic environment class `film_env.py` is defined. It is a custom gym environment,and it interacts with the C++ simulation that is defined in `simulation_solver`.
- `model`, where the script to retrieve models is defined. These models are imported from the library stable_baselines, and only the PPO2 implementation has been used in the paper.
- `results`, where previously trained models are stored along with the parameters used for the training, and tensorboard logs to have insights on how the model performs during training


## You need an environment in which you can run the code
Because the simulation is built from scratch in C++ and linked with the Python API with the library Boost.Python, you need to have the library installed for the simulation to run.

The easiest way is to run the code in a docker container, built from a Ubuntu 18.04 image. This way you can run the code whether you are on Windows, MacOS or Linux as long as you can run docker.

### Run it with docker (recommended)

Docker is a convenient tool to use containers. If you're not familiar with the concept, everything you need to run code from this project will be inside the container. In this case, it will be an Ubuntu 18.04 distribution with Python3 and other necessary packages.
The details of how I built my docker image from `ubuntu:latest` are in [this repository](https://github.com/vbelus/docker_fluid_film). You can find all these steps in the `Dockerfile`.

- Once you can use Docker, get the image :
  - from the `Dockerfile` by running the following command in the same directory as the `Dockerfile` (this can take several minutes or more depending on your internet connection, and the docker image will take 2.4GB on your disk):
  ```shell
  $ docker build . -t falling-liquid-film:latest
  ```
  - by downloading it from the following adress : https://folk.uio.no/jeanra/Research/falling-liquid-film_latest.tar.gz

- You can then run a container from this image with the following command :
```shell
$ docker run -it falling-liquid-film:latest
```

- You should now be in the `gym-film` folder, ready to launch trainings XX: make display possible

### Run it with Ubuntu

XX: fill this section, but basically same as Dockerfile

## How do I train some models ?

### Execute the main script

The basic command is `python3 train.py` but you will need to add some flags :
- `--train` or `-t` to train your model
- `--render` or `-r` if you want to render the simulation

If you use this flag with `-t`, it will render during the training as well. You can change how often the rendering is done with the `RENDER_PERIOD` parameter
- `--training_name` or `-tn` is the name of the training, default value is "test"
- `--load` or `-l` followed by an integer `n` to load a trained model. This will look for a model trained for n episodes under the training name you specified
- `--load_best` or `-lb` will load the "best" model under the training name you specified, "best" meaning the model trained on k episodes, where the maximum episode reward of the entire training was obtained during episode k
- `--port` or `-p`, followed by an integer between 1024 and 49151, which will be the port used when using the multi-environment method (Method M3 in the paper).

If you want to train a model with the default parameters and render it, you can do it with `python train.py --train --render --training_name my-first-training`

### Change the parameters of the training
All the parameters relevant to the training and simulation are in `param.py`. You will have to change the parameters in this file.

We will use the following syntax: `parameter` (type, typical low value - typical high value)

Some important parameters here are :
- The number of jets: `n_jets` (int, 1 - 20)
- The position of the jets: `jets_position` (array) OR `position_first_jet` (float) and `space_between_jets` (float, 5 - 20)
- The maximum power of the jets: `JET_MAX_POWER` (float, 1 - 15) and their width: `JET_WIDTH` (float, 2 - 10)
-> As it is done, the action of the agent is always between `-1` and `1` and is later normalized and multiplied by `JET_MAX_POWER` to be applied in the simulation. The default values are letting the jets modify q with the same amplitude as the waves naturally forming, so it should be enough for control. 
- The size of the observation `size_obs` (float, 10 - 40), input of the agent, and the size of the observation on which the reward is calculated `size_obs_to_reward` (float, 5 - 20)
- The non-dimensional duration of one episode `simulation_time` (float, 5 - 20)
- The duration of the simulation before we begin any training `initial_waiting_time` (float). Default value is 200, letting the simulation get to a converged state before we do any training (the state is stored and not computed each time)
- `simulation_step_time` is the non-dimensional duration of one step of the simulation. In one such step, we do `n_step` steps of the numerical scheme
- Whether to render the simulation with matplotlib with `render` (bool). If True, render it every `RENDER_PERIOD` (int) environment steps

## That would be nice if I could visualize all that in a jupyter notebook
And that is exactly what you can do here: [**LINK TO THE NOTEBOOK REPO**](https://github.com/vbelus/drl-fluid-film-notebook)
**You need Docker to run this**

This notebook will demonstrate how the three implemented methods build on top of each other, and you will be able to look at what the different parameters do on the simulation interactively. 

I made it for a course on Deep Reinforcement Learning at my school (Mines ParisTech, PSL University).
