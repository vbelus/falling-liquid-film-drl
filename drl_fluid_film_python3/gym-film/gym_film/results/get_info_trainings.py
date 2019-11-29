import os
import pickle
import numpy as np
import param
dir = "./to_plot/reward_space_100"
rmnan = True


def remove_nans(arr):
    # remove nans from array, and return their indexes
    nans, x = (np.isnan(arr)) + (arr <= param.nan_punition), lambda z: z.nonzero()[0]
    x_nans = x(nans)
    y_nans = np.interp(x_nans, x(~nans), arr[~nans])
    arr[nans] = y_nans
    return x_nans, y_nans


for filename in sorted(os.listdir(dir)):
    if not ".obj" in filename:
        continue
    Y_data_one_training = []
    with open(os.path.join(dir, filename), 'rb') as file_one_training:
        while True:
            try:
                Y_data_one_episode = pickle.load(file_one_training)
            except Exception as e:
                print(e)
                print("In training: ", filename)
                break

            Y_data_one_training.append(Y_data_one_episode)

    # we make it a single array
    Y_data_one_training = np.concatenate(Y_data_one_training)

    if rmnan:
        # we remove the NaN
        x_nans, y_nans = remove_nans(Y_data_one_training)

    # we print some info
    print('number of points: ', len(Y_data_one_training),
          (', number of nans: {}'.format(len(x_nans)) if rmnan else ''))
    print('mean :', np.mean(Y_data_one_training))
    print('std, ', np.std(Y_data_one_training))
    print('max: ', np.max(Y_data_one_training),
          ', min: ', np.min(Y_data_one_training))
    n_points_to_show = 10
    print('reduced array: ', Y_data_one_training[np.linspace(
        0, len(Y_data_one_training)-1, n_points_to_show, dtype='int')])
    print('\n')
