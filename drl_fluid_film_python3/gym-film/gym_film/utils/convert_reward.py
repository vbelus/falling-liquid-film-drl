'''
In the multienv method, we have n_jets different rewards at each step
However, we need to compare the different methods with a plot showing a single reward per step

Input - a list of rewards' lists: [reward_jet_1, rewards_jet_2, ..., reward_jet_n]

                           with: reward_jet_k = [reward_step1, reward_step2, ..., reward_step400] 
                           in the default case of 20s episodes with 0.05s steps

Output - a single list of reward: [reward_step1, reward_step2, ..., reward_step400] 
         with mathematically the same reward as other methods for the same observation
'''

import numpy as np
import param


def to_single_reward(list_rewards):
    list_rewards = np.array(list_rewards)
    return 1 - param.reward_param * np.sqrt((1/param.n_jets)*sum([((1-list_rewards[k])/param.reward_param)**2 for k in range(len(list_rewards))]))
