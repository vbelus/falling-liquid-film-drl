import param
import numpy as np
from gym import spaces

# Takes number of jets, can output
#   - the observation_space that the environment needs
#   - the observation as a function of the jets' position and system state


class Observation():
    def __init__(self, n_jets):
        n_measures = int(param.size_obs / param.dx)
        obs_points = np.arange(
            0, int(param.cut_obs_before_jet * n_measures))
        # We take input til the right of th jet (where it can still control something)
        obs_points += param.JET_WIDTH//2
        # we place the sensors to be on the left of the jet
        obs_points -= n_measures

        self.obs_points = obs_points
        self.n_points = len(obs_points)

        self.n_jets = n_jets
        self.observation_space = self._get_observation_space()
        self._obs = np.empty(self.observation_space.shape)

    def _get_observation_space(self):
        if self.n_jets == 1:
            obs_space_shape = (self.n_points, 2)
        else:
            obs_space_shape = (self.n_jets, self.n_points, 2)
        low_observation = -param.max_q * np.ones(obs_space_shape)
        high_observation = param.max_q * np.ones(obs_space_shape)
        return spaces.Box(low=low_observation,
                          high=high_observation,
                          dtype=np.float64)

    def get_observation(self, system_state, jets_position):
        if self.n_jets == 1:
            # we observe h
            self._obs[:, 0] = self._process_hq(
                system_state[0], jets_position, normalize_value=param.normalize_value_h)
            # we observe q
            self._obs[:, 1] = self._process_hq(
                system_state[1], jets_position, normalize_value=param.normalize_value_q)
            return self._obs
        else:
            # we observe h
            self._obs[:, :, 0] = self._process_hq(
                system_state[0], jets_position, normalize_value=param.normalize_value_h)
            # we observe q
            self._obs[:, :, 1] = self._process_hq(
                system_state[1], jets_position, normalize_value=param.normalize_value_q)
            return self._obs

    def _process_hq(self, hq, jets_position, normalize_value=1):
        if self.n_jets == 1:
            # we set the obs points where the jet is
            obs_points = self.obs_points+jets_position
        else:
            # we set the obs points where the jets are
            obs_points = np.array(
                [self.obs_points+one_jet_position for one_jet_position in jets_position])
        # we process the obs, set mean at 0 and clip it
        return np.clip(normalize_value*(hq[obs_points] - param.hq_base_value),
                       a_min=-param.threshold_hq,
                       a_max=param.threshold_hq)
