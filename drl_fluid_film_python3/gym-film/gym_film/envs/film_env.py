import gym
from gym import spaces
import numpy as np
import math

from gym_film.envs import render
from gym_film.envs.simulation_solver.simulation import Simulation

from gym_film.envs.observation.observation import Observation
from gym_film.envs.reward.reward import Reward

import param
import logging
logger = logging.getLogger(__name__)

# A custom openAI gym environment for film fluid flow control with drl
# %%


class FilmEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, jets_position=None, n_jets=param.n_jets, plot_jets=True, **kwargs):
        super(FilmEnv, self).__init__()
        self.jet_position = param.jets_position if jets_position is None else jets_position
        self.n_jets = n_jets
        self.rendering = kwargs["render"]

        self.simulation = Simulation()
        self.system_state = self.simulation.get_system_state()
        self.current_epoch = 0

        if param.monitor_reward:
            self._reward_list = []

        # Actions are % of max blow in the N jets:
        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(self.n_jets,),
                                       dtype=np.float64)

        # observations are values of h and q on the left of the jets
        self.Ob = Observation(self.n_jets)
        self.observation_space = self.Ob.observation_space

        # rewards are calculated on the right of the jetss
        self.R = Reward(self.n_jets)

        # Here to make sure plt.close and initialization of plot
        # happen only once, at the first rendering
        self.reward = 0  # for render class
        self.t = 0  # for render class
        self.first_render = True  # for render class
        self.plot_jets = plot_jets
        self.render_step = 0  # for render class

    def reset(self, hard_reset=False):
        if hard_reset:
            self.simulation.reset()
        self.system_state = self.simulation.get_system_state()

        self.current_epoch += 1
        self.current_step = 0

        obs = self._next_observation()
        return obs

    def step(self, action):
        # Execute one time step within the environment
        self.jet_power = action
        self._take_action(action)

        # We observe our environment
        obs = self._next_observation()
        # a, b, c, d = np.max(obs[:,:,0]),
        #              np.min(obs[:,:,0]),
        #              np.max(obs[:,:,1]), np.min(obs[:,:,1])
        # if a>1 or c>1 or b<-1 or d<-1:
        #     print("ohla oh OH")
        #     print("max h:, ", a)
        #     print("min h:, ", b)
        #     print("max q:, ", c)
        #     print("min q:, ", d)

        # We get the reward
        reward = self._next_reward()
        self.reward = reward

        # eventually render
        if (self.rendering):
            self.render_step += 1
            if self.render_step % param.RENDER_PERIOD == 0:
                self.render()

        if param.monitor_reward:
            self._reward_list.append(reward)

        if math.isnan(reward):
            print("Oups, reward is nan")
            print(self.system_state, obs)
            obs = self.reset(hard_reset=True)
            return obs, param.nan_punition, True, {}

        # check if we have finished simulation
        self.current_step += 1
        done = self.current_step >= param.nb_timestep_per_simulation

        return obs, reward, done, {}

    # Take action and update the state of the environment
    def _take_action(self, action):
        self.jets_power = action  # important for rendering

        # get actual jet_power by multiplying by max power
        jet_power = action * param.JET_MAX_POWER

        # make simulation evolve with new jet_power
        self.simulation.next_step(jet_power)

        # get system state
        self.system_state = self.simulation.get_system_state()

        # get time
        self.t = self.simulation.get_time()

    def _next_observation(self):
        return self.Ob.get_observation(self.system_state, self.jet_position)

    def _next_reward(self):
        return self.R.get_reward(self.system_state, self.jet_position)

    def _no_action_step(self):
        action = self.action_space.sample()
        action.fill(0)
        return self.step(action)

    def _full_power_action_step(self):
        action = self.action_space.high
        return self.step(action)

    def render(self, mode='human', close=False, compare_without_control=False,
               plot_sensor=False, sensor_position=None):
        if self.first_render:
            self.film_render = render.FilmRender(
                self, PLOT_JETS=self.plot_jets)
            self.first_render = False

        else:
            self.film_render.update_plot()

    # get reward list - only at the end of an epoch
    def get_epoch_reward(self):
        print(self.current_step)
        reward_list = self._reward_list.copy()
        self._reward_list = []
        return reward_list

# %%
