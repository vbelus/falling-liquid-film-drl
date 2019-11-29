import param
import numpy as np

# Takes number of jets, can output
#   - the reward as a function of the jets' position and system state


class Reward():
    def __init__(self, n_jets, size_obs_to_reward=param.size_obs_to_reward):
        self.n_jets = n_jets

        n_measures_to_reward = int(size_obs_to_reward / param.dx)
        obs_points_to_reward = np.arange(0, n_measures_to_reward)
        obs_points_to_reward -= param.JET_WIDTH//2
        self.obs_points_to_reward = obs_points_to_reward
        self.n_points_to_reward = len(self.obs_points_to_reward)

        if self.n_jets == 1:
            obs_to_reward_space_shape = (self.n_points_to_reward)
        else:
            obs_to_reward_space_shape = (self.n_jets, self.n_points_to_reward)

        self._obs_to_reward = np.empty(obs_to_reward_space_shape)

    def _get_observation_to_reward(self, system_state, jets_position):
        # we observe h
        self._obs_to_reward[:] = self._process_hq(
            system_state[0], jets_position)

    def _process_hq(self, hq, jets_position):
        if self.n_jets == 1:
            # we set the obs_to_reward points where the jet is
            obs_points_to_reward = self.obs_points_to_reward+jets_position
        else:
            # we set the obs_to_reward points where the jets are
            obs_points_to_reward = np.array(
                [self.obs_points_to_reward+one_jet_position for one_jet_position in jets_position])
        # we process the obs, set mean at 0
        return param.obs_param*(hq[obs_points_to_reward] - param.hq_base_value)

    def get_reward(self, system_state, jets_position):
        method = 'distance_to_0'
        self._get_observation_to_reward(system_state, jets_position)
        flattened_obs_to_reward = self._obs_to_reward.flatten()

        if method == 'distance_to_0':
            reward = 1 - (param.reward_param
                        * np.linalg.norm(flattened_obs_to_reward)
                        / np.sqrt(len(flattened_obs_to_reward)))
        elif method == 'std':
            reward = 1 - np.std(flattened_obs_to_reward)
        elif method == 'distance_to_x':
            x = 1
            reward = 1 - (param.reward_param
                        * np.linalg.norm(x - flattened_obs_to_reward)
                        / np.sqrt(len(flattened_obs_to_reward)))
        return reward

class RewardBase(Reward):
    def __init__(self, n_jets, size_obs_to_reward):
        super(RewardBase, self).__init__(n_jets, size_obs_to_reward=size_obs_to_reward)
    def get_reward(self, system_state, jets_position):
        self._get_observation_to_reward(system_state, jets_position)
        flattened_obs_to_reward = self._obs_to_reward.flatten()
        reward = 1 - (param.reward_param
                      * np.linalg.norm(flattened_obs_to_reward)
                      / np.sqrt(len(flattened_obs_to_reward)))
        return reward
