if __name__ == '__main__':
    import warnings
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    # imports
    import pickle
    import os
    import filecmp
    import shutil
    import argparse
    # from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines.common.vec_env import SubprocVecEnv
    import gym
    import numpy as np

    # imports from the project directory
    from gym_film.model import get_model
    import param
    from gym_film.utils.convert_reward import to_single_reward
    from gym_film.utils import save_name
    from gym_film.envs import make_env

    ##################################### Adding arguments for command line #####################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("-tn",
                        "--training_name",
                        help="name that will show on your saved models and tensorboard logs")
    parser.add_argument(
        "-l", "--load", help="load model with certain number of epochs")
    parser.add_argument("-lb",
                        "--load_best", help="load the best model for a certain training name", action="store_true")
    parser.add_argument("-r", "--render", help="to render",
                        action="store_true")
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-p", "--port", help="port to use")
    parser.add_argument("--ls", help="list models", action="store_true")
    parser.add_argument("-tb", "--tensorboard", action="store_true")

    args = parser.parse_args()
    #############################################################################################################################################

    ##################################### Training's info and parameters ########################################################################
    # Model name
    model_name = "PPO2"

    # Environment ID
    env_id = 'film-v0'

    # Training's name - default: 'test'
    if not args.training_name:
        training_name = 'test'
        print('No training name specified, default : {}'.format(training_name))
    else:
        training_name = args.training_name

    # Dumping directories (models, tensorboard)
    tensorboard_log = "./gym_film/results/tensorboard_logs/"+training_name
    models_dir = "./gym_film/results/models/"+training_name+"/"
    best_model_dir = models_dir+"best_model/"

    best_mean_reward = None

    # if no training_name, delete previous test models
    if save_name.get_training_name(models_dir) == "test":
        if os.path.exists(models_dir):
            shutil.rmtree(models_dir)

    # create the directory to save models
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
        # save params
        shutil.copy('param.py', models_dir)

    # if already existing directory, check that the params are the same
    else:
        if not filecmp.cmp('param.py', models_dir+'param.py', shallow=False):
            print("Your parameters and this training's parameters are not identical.")
            user_input = input(
                "Use training's parameters (it will erase your parameters) ? [y/n]")
            if user_input in ["Y", "y", "Yes", "yes"]:
                shutil.copy(models_dir+"param.py", "./")
            elif user_input in ["N", "n", "No", "no"]:
                pass
            else:
                raise Exception()
    #############################################################################################################################################

    ################################################ Getting the environment ####################################################################
    # The algorithms require a vectorized environment to run

    '''
        Two possible methods:
        - "1env_njet": the model takes observations from all the jets at once, and control them all at once
                    we use multiple environments, each one using a vcpu, to accelerate the training
                    n_cpus = n_envs = n_simulations != n_jets

        - "1env_1jet": the model takes observations from one jet and control only that one jet
                    we use multiple environments on one simulation, to be more resource efficient
                    n_cpus = n_envs = n_jets != n_simulations = 1
    '''

    method = param.method
    n_jets = param.n_jets
    jets_position = param.jets_position

    envs = make_env.make_env(
        method, n_jets, jets_position, port=args.port, render=args.render)

    # We now have a vector of several environments that we can run in parallel
    env = SubprocVecEnv(envs)
    ##############################################################################################################################################

    ################################################ Getting the model ###########################################################################
    '''
    Load model - get the model with input number of training episodes for a certain training
    Best model - get the model with best avg reward on his episode
    New model - new model with random policy at first
    '''
    assert not (args.load and args.load_best)
    # load from a certain number of epochs
    model_path = None
    if args.load:
        mode = "load_model"
        model_path = models_dir + save_name.SaveName(dic={"number of epochs": args.load,
                                                          "model name": model_name,
                                                          "environment id": env_id}).save_name

    # load the best model
    elif args.load_best:
        mode = "best_model"

    # create new model
    else:
        mode = "new_model"
    # we retrieve the model
    model, nb_epoch = get_model.get_model_PPO2(mode,
                                               env,
                                               model_name,
                                               env_id,
                                               tensorboard_log,
                                               best_model_dir=best_model_dir,
                                               model_path=model_path,
                                               policy_name=param.policy_name,
                                               tensorboard_integration=args.tensorboard)

    # when saving the model, the saving name includes - number of epochs, model name (PPO2) anv Env ID
    S = save_name.SaveName(dic={"number of epochs": nb_epoch,
                                "model name": model_name,
                                "environment id": env_id})
    model_save_name = S.save_name
    ##############################################################################################################################################

    # the total number of steps in training depends on the number of parallel environments
    # because we want the full simulation time to be constant
    total_nb_timesteps = param.total_nb_timesteps * len(envs)

    ################################################ If there's training #########################################################################
    if args.train:

        # subdirectory where the unique best model is stored
        if not os.path.exists(best_model_dir):
            os.mkdir(best_model_dir)

        # open file where we will dump the rewards
        file_reward_list = open(models_dir + training_name + '.obj', 'wb')

        def callback(_locals, _globals):
            global nb_epoch
            global best_mean_reward
            global file_reward_list
            global S
            S.add_epoch(1)

            print("Epoch number {}".format(nb_epoch))
            print("We are in following training: {}".format(training_name))

            if param.monitor_reward:

                # the following method should return an array with the rewards of the episode at each step
                reward_list = _locals["self"].env.env_method(
                    "get_epoch_reward")

                if param.method == "1env_njet":
                    # if we launched multiple trainings, we have a list of reward lists here, so we take the first element
                    single_reward_list = reward_list[0]

                if param.method == "1env_1jet":
                    # at this point, we have a list of the lists of rewards of each jet-environment
                    # we convert it to a single list of rewards, with the right formula
                    single_reward_list = to_single_reward(reward_list)

                pickle.dump(single_reward_list, file_reward_list)

                mean_reward = np.mean(single_reward_list)
                print('Mean episode reward : ', mean_reward)

                # We check if the model at this step is better than the previous best

                # TODO - now if we launch a training from a preexisting model,
                #        the best model will immediately be replaced by the new,
                #        because of the following condition. The best reward
                #        can be added to the model object to fix that
                if best_mean_reward == None:
                    best_mean_reward = mean_reward
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    shutil.rmtree(best_model_dir)
                    os.mkdir(best_model_dir)
                    _locals["self"].save(best_model_dir + str(nb_epoch) + "epochs")

            # We also save models every n epochs
            if nb_epoch % param.save_every_n_epoch == 0:
                model_save_name = S.save_name
                _locals["self"].save(models_dir+model_save_name)

            # we go to the next epoch
            nb_epoch += 1

        # We learn here
        model.learn(total_nb_timesteps, callback=callback)

    ##############################################################################################################################################

    render_no_jet = False
    ################################################ If there's rendering ########################################################################
    if args.render:
        if method == "1env_njet":
            # we don't need n_cpu parallel environments for rendering in first method
            env = gym.make('gym_film:film-v0', n_jets=n_jets,
                           jets_position=jets_position, render=True)
            # env = DummyVecEnv(envs)

        # We render a no_jet environment as well
        no_jet_env = gym.make('gym_film:film-v0', n_jets=n_jets,
                              jets_position=jets_position, render=True)
        render_total_timesteps = total_nb_timesteps

        if render_no_jet:
            no_jet_obs = no_jet_env.reset()
        obs = env.reset()

        for i in range(render_total_timesteps):
            if render_no_jet:
                # render no jet env
                no_jet_obs, no_jet_rewards, no_jet_done, no_jet_info = no_jet_env._full_power_action_step()

            # render jet env
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
