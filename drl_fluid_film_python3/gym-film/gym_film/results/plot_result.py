import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np
import param
from scipy import interpolate

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--njets', type=int, help='n_jets of trainings to compare')
parser.add_argument('-d', '--dir', type=str, help='directory where we will fetch results')
parser.add_argument('--cpu_time', action="store_true")
args = parser.parse_args()

matplotlib.rcParams.update({'font.size': 15})

to_plot = [{"dir_name": "base",
            "legend": 'base',
            "x_step": 1,
            "color": "blue"},
            {"dir_name": "reward_space_100",
            "legend": r'reward domain ×10',
            "x_step": 1,
            "color": "red"},
            {"dir_name": "rew_is_std_ultimate",
            "legend": 'reward using std',
            "x_step": 1,
            "color": "green"},
            {"dir_name": "sst_is_dt_with_same_nbofsst_per_ep",
            "legend": '1 action/solver step',
            "x_step": 1,
            "color": "orange"}]

# we get the full path for each training to plot
for dic in to_plot:
    if args.njets:
        dic['dir_name'] = dic.get('dir_name') + str(args.njets) + 'jet'
    dic['dir_name'] = os.path.join(args.dir, dic.get('dir_name'))

if args.njets:
    title = "Reward over number of steps, with {} jet".format(args.njets)
    if args.njets > 1:
        title += 's'

######################## X #######################
reduce_x = 1000

X_label = X_label = r'Number of {} ($\times {}$)'.format(('simulation steps' if args.cpu_time else 'actions'), reduce_x)
Y_label = "Reward"

X_min = 0
X_max = 300000

if args.njets == 1:
    X_max = 50000
# if args.njets == 10:
#     X_max = 170000
# if args.njets == 1:
#     X_max = 70000
##################################################

######################## Y #######################
Y_minmax_auto = False
Y_min = -1.1
Y_max = 1
##################################################

avg = False
rmnan = True

smooth_every_n_step = (X_max-X_min)//100 # if larger than 1, will take the avg of the Y_mean every few steps (to have a render more smooth)
linewidth_one_training = 0.02 #0.02
linewidth_mean = 2

def remove_nans(arr):
    # remove nans from array, and return their indexes
    nans, x= (arr<=param.nan_punition) + (np.isnan(arr)), lambda z: z.nonzero()[0]
    x_nans = x(nans)
    y_nans = np.interp(x_nans, x(~nans), arr[~nans])
    arr[nans]= y_nans
    return x_nans, y_nans

def smooth(X, Y, smooth_every_n_step):
    # we smooth it
    n_points_in_smooth = len(X)//smooth_every_n_step
    smooth_Y = np.empty(n_points_in_smooth)
    for i in range(n_points_in_smooth):
        smooth_Y[i] = np.mean(Y[smooth_every_n_step*i:smooth_every_n_step*(i+1)])
    smooth_X = X[np.linspace(0, len(X)-1, n_points_in_smooth, dtype='int')]
    return smooth_X ,smooth_Y

# for each training
for iteration in to_plot:

    # we define our X
    X_data = np.arange(X_min, X_max, iteration.get("x_step"))
    X_data = X_data / reduce_x
    n_points = len(X_data)

    # here we will store the multiple lists of rewards, that we will average later
    Y_data = []

    path = iteration.get("dir_name")
    color = iteration.get("color")
    # get the different iterations of same training
    for filename in os.listdir(path):
        try:
            # this should be a pickle file we have to load many times until we have the full data
            # data at the end is the reward of our simulation at each step of the environment
            Y_data_one_training = []
            with open(os.path.join(path,filename), 'rb') as file_one_training:
                while True:
                    try:
                        Y_data_one_episode = pickle.load(file_one_training)
                    except:
                        break

                    if avg:
                        # we take the avg on the episode
                        Y_data_one_episode = [np.mean(Y_data_one_episode)]
                    Y_data_one_training.append(Y_data_one_episode)

            # we make it a single array
            Y_data_one_training = np.concatenate(Y_data_one_training)[:n_points]

            if rmnan:
                # we remove the NaN
                x_nans, y_nans = remove_nans(Y_data_one_training)
                x_nans = X_data[x_nans]

                # plot points where there were nans
                plt.scatter(x_nans, y_nans, color=color, s=0.3)
            
            # we plot it 
            plt.plot(X_data[:len(Y_data_one_training)], Y_data_one_training, linewidth=linewidth_one_training, linestyle='--', color=color)

            # we add it to the data
            Y_data.append(Y_data_one_training)
        except:
            raise Exception("Error in following file:", filename)

    # we take the average after calculating its length (if we have different iterations data with different lengths)
    n_points_in_mean = min([len(Y) for Y in Y_data])    
    Y_data_mean = np.mean([Y[:n_points_in_mean] for Y in Y_data], axis=0)
    X_data_mean = X_data[:len(Y_data_mean)]

    # we smooth it
    X_data_mean, Y_data_mean = smooth(X_data_mean, Y_data_mean, smooth_every_n_step//iteration.get('x_step'))

    # we plot it 
    plt.plot(X_data_mean, Y_data_mean, linewidth=linewidth_mean, color=color, label=iteration.get("legend"))
    plt.legend()

#plt.title(title)

plt.xlabel(X_label)
plt.ylabel(Y_label)

plt.xlim(X_min/reduce_x, X_max/reduce_x)
if not Y_minmax_auto:
    plt.ylim(Y_min,Y_max)


if rmnan:
    plt.scatter([], [], color='k', s=3)


plt.legend(loc="lower left")

plt.tight_layout()

#fullscreen
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

# if args.njets !=1:
#     plt.gca().get_legend().remove()

save=1
if save:
    plt.savefig(os.path.join('/home/vbelus/Pictures/plot_fluidfilmpaper','figure_compare_parameters'+('_cpu_time' if args.cpu_time else ''),'figure_nice'+str(args.njets)+'jet_.png'), format='png')
    plt.savefig(os.path.join('/home/vbelus/Pictures/plot_fluidfilmpaper','figure_compare_parameters'+('_cpu_time' if args.cpu_time else ''),'figure_nice'+str(args.njets)+'jet_.pdf'), format='pdf')

show = 0
if show:
    #show
    plt.show()