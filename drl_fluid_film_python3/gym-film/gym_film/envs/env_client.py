import socket
from gym_film.envs.echo_server import EchoServer
import time


import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

from gym_film.envs import render
from gym_film.envs.observation.observation import Observation


import param


# A custom openAI gym environment for film fluid flow control with drl

import logging
logger = logging.getLogger(__name__)

# %%


class FilmEnvClient(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 verbose=1,
                 buffer_size=262144,
                 timing_print=False,
                 **kwargs
                 ):

        super(FilmEnvClient, self).__init__()

        self.jet_position = kwargs["jets_position"]
        self.rendering = kwargs["render"]

        if param.monitor_reward:
            self._reward_list=[]

        # actions are proportion of maximum_jet_power for the one jet of this env
        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(1,),
                                       dtype=np.float64)

        # observations are values of h and q on the  left of the jet
        self.Ob = Observation(1)
        self.observation_space = self.Ob.observation_space

        # Here to make sure plt.close and initialization of plot happen only once, at the first rendering
        self.first_render = True
        self.current_epoch = 0

        ###################################################################
        # socket
        self.port = kwargs["port"]
        self.host = kwargs["host"]
        # misc
        self.verbose = verbose
        self.buffer_size = buffer_size

        # start the socket
        self.valid_socket = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # connect to the socket
        while True:
            try:
                self.socket.connect((self.host, self.port))
                break
            except socket.error:
                print('Connection failed, trying again...')
                time.sleep(1)
        if self.verbose > 0:
            print('Connected to {}:{}'.format(self.host, self.port))

        # now the socket is ok
        self.valid_socket = True

        self.current_epoch = 0
        self.current_step = 0

    def reset(self, hard_reset=False):
        # Updating episode and step numbers
        self.current_epoch += 1
        self.current_step = 0

        # perform the reset or keep the simulation going
        if hard_reset:
            _ = self.communicate_socket("RESET", 1)
        # get the obs
        _, init_obs = self.communicate_socket("OBS", self.jet_position)

        if self.verbose > 1:
            print("reset done; init_state:")
            print(init_obs)

        return(init_obs)

    def step(self, action):
        # send the control message
        self.communicate_socket("CONTROL", [self.jet_position, action])

        # obtain the next observation
        _, next_obs = self.communicate_socket("OBS", self.jet_position)

        # check if episode is done
        _, done = self.communicate_socket("DONE", 1)

        # get the reward
        _, reward = self.communicate_socket("REWARD", self.jet_position)

        _, reward_base = self.communicate_socket("REWARD_BASE", self.jet_position)

        # and store it
        if param.monitor_reward:
            self._reward_list.append(reward_base)

        # now we have done one more step
        self.current_step += 1

        if self.verbose > 1:
            print("execute performed; state, terminal, reward:")
            print(next_obs)
            print(done)
            print(reward)

        return next_obs, reward, done, {}

    def communicate_socket(self, request, data):
        """Send a request through the socket, and wait for the answer message.
        """
        to_send = EchoServer.encode_message(
            request, data, verbose=self.verbose)
        self.socket.send(to_send)

        received_msg = self.socket.recv(self.buffer_size)

        request, data = EchoServer.decode_message(
            received_msg, verbose=self.verbose)

        return(request, data)

    def get_epoch_reward(self):
        print(self.current_step)
        reward_list = self._reward_list.copy()
        self._reward_list=[]
        return reward_list
