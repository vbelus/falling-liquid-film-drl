import socket
import pickle
import threading
import numpy as np
import os
import math


from gym_film.envs.echo_server import EchoServer
from gym_film.envs import render

from gym_film.envs.observation.observation import Observation
from gym_film.envs.reward.reward import Reward, RewardBase

import param


class FilmEnvServer(EchoServer):

    def __init__(self,
                 simulation,
                 host=None,
                 port=12230,
                 buffer_size=262144,
                 verbose=1,
                 render=False):

        # tensorforce_environment should be a ready-to-use environment
        # host, port is where making available

        self.verbose = verbose
        self.done = False
        self.current_step = 0
        self._reward_punition = False

        self.rendering = render

        self.reward = {param.jets_position[k]
            : 0 for k in range(param.n_jets)}  # for render class
        self.t = 0 # for render class
        self.first_render = True  # for render class
        self.render_step = 0  # for render class

        self.simulation = simulation
        self.system_state = np.array(self.simulation.get_system_state())

        self.Ob = Observation(1)
        self.R = Reward(1)
        self.R_base = RewardBase(1, 10)

        self.action = {param.jets_position[k]
            : None for k in range(param.n_jets)}
        self.action_lock = threading.Lock()
        self.get_obs_event = threading.Event()

        self.buffer_size = buffer_size
        if host is None:
            host = socket.gethostname()
        self.host = host
        self.port = port
        # set up the socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))

        EchoServer.__init__(self, self.verbose)

        # start listening
        self.listen()
        self.sock.close()

    def listen(self):
        self.sock.listen(1)
        while True:
            print("Listening for incoming connections")
            client, address = self.sock.accept()
            if self.verbose > 1:
                print('Got connection from {}'.format(address))
            client.settimeout(60)
            threading.Thread(target=self.listen_to_client,
                             args=(client, address)).start()

    def listen_to_client(self, client, address):
        while True:
            try:
                if self.verbose > 1:
                    print('[Waiting for request...]')
                message = client.recv(self.buffer_size)
                # this is given by the EchoServer base class
                response = self.handle_message(message)
                client.send(response)
            except Exception as e:
                print(e)
                client.close()
                return False

    def RESET(self, data):
        self.simulation.reset()
        self.system_state = self.simulation.get_system_state()
        self.current_step = 0
        return(1)  # went fine

    # data: jet_position
    # returns: observation for input jet
    def OBS(self, data):
        obs = self._get_obs(data)
        return obs

    def DONE(self, data):
        done = self._get_done()
        return done

    def REWARD(self, data):
        reward = self._get_reward(data)
        return reward

    def REWARD_BASE(self, data):
        jet_position = data
        if self._reward_punition:
            reward_base = param.nan_punition
        else:
            reward_base = self.R_base.get_reward(self.system_state, jet_position)
        return reward_base

    def CONTROL(self, data):
        self._set_jet(data)
        return(1)  # went fine

    def _get_done(self):
        if np.isnan(np.sum(self.system_state)):
            self._reward_punition = True
            print("Numerical scheme collapsed, punishing reward and resetting simulation")
            return 1
        return self.current_step >= param.nb_timestep_per_simulation

    def _get_obs(self, jet_position):
        # we get q, and h
        return self.Ob.get_observation(self.system_state, jet_position)

    def _get_reward(self, jet_position):
        if self._reward_punition:
            reward = param.nan_punition
        else:
            reward = self.R.get_reward(self.system_state, jet_position)
        self.reward[jet_position] = reward
        return reward

    def _set_jet(self, jet_info):
        # format: [[150], 0.593] for action 0.593 at jet whose position is 150
        jet_position, action = jet_info
        action = action[0]

        # later, wait for all the actions to be set
        self.get_obs_event.clear()
        # we shouldnt punish reward anymore
        if self._reward_punition:
            self.RESET(1)
            self._reward_punition = False

        with self.action_lock:  # we manipulate the array of jet powers sequentially
            self.action[jet_position] = action
            if not None in self.action.values():  # if all actions have been given

                # passing the actions to the simulation
                self.jets_power = [self.action.get(
                    jet_position) for jet_position in sorted(self.action.keys())]
                jet_power = np.array(self.jets_power)*param.JET_MAX_POWER
                self.simulation.next_step(jet_power)

                # get the state of the simulation
                self.system_state = self.simulation.get_system_state()

                # potentially render
                if (self.rendering):
                    if ((self.render_step % param.RENDER_PERIOD == 0) or (self.render_step==0)):
                        self.render()
                    self.render_step += 1
                self.t = self.simulation.get_time()
                
                # add one step and check done
                if self._get_done():
                    self.current_step = 0
                self.current_step += 1

                # we  reset the actions
                self.action = {
                    param.jets_position[k]: None for k in range(param.n_jets)}
                
                if param.true_reset_every_n_episodes == True:
                    # true reset from time to time
                    if int(self.t)%400==0:
                        self.RESET(1)

                # we let the other threads go on
                self.get_obs_event.set()

        # wait for all the actions to be set
        # only if the environments are actually parallel
        if param.is_dummy_vec_env:
            pass
        else:
            self.get_obs_event.wait()

    def render(self, mode='human', close=False, compare_without_control=False, plot_sensor=False, sensor_position=None):
        if self.first_render == True:
            self.film_render = render.FilmRender(self)
            self.first_render = False
        else:
            self.film_render.update_plot()
