from stable_baselines import PPO2
import param
import os

import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)

def get_model_PPO2(mode, env, model_name, env_id, tensorboard_log, best_model_dir=None, model_path=None, policy_name=None, tensorboard_integration=True):
    # policy - "mlp", "cnn"
    if policy_name == "mlp":
        from gym_film.model.custom_mlp import CustomPolicy
        policy = CustomPolicy
    if policy_name == "cnn":
        from gym_film.model.custom_cnn import CustomCnnPolicy
        policy = CustomCnnPolicy
    if policy_name == "mlp_shared":
        from gym_film.model.custom_shared_mlp import CustomPolicy
        policy = CustomPolicy

    # mode - "new_model", "load_model", "best_model"
    if mode == "new_model":
        if tensorboard_integration:
            model = PPO2(policy, env=env, n_steps=param.nb_timestep_per_simulation,
                        verbose=1, tensorboard_log=tensorboard_log)
        else:
            model = PPO2(policy, env=env, n_steps=param.nb_timestep_per_simulation,
                        verbose=1)
        nb_epoch = 0

    elif mode == "best_model":
        best_model = best_model_dir + os.listdir(best_model_dir)[0]
        print('Loading model: {}'.format(best_model))
        if tensorboard_integration:
            model = PPO2.load(best_model, env=env, tensorboard_log=tensorboard_log)
        else:
            model = PPO2.load(best_model, env=env)
        nb_epoch = int(best_model.split('epochs')[0].split('/')[-1])
        print(nb_epoch)

    elif mode == "load_model":
        print('Loading model: {}'.format(model_path))
        if tensorboard_integration:
            model = PPO2.load(model_path+".zip", env=env, tensorboard_log=tensorboard_log)
        else:
            model = PPO2.load(model_path+".zip", env=env)
        nb_epoch = int(model_path.split('epochs')[0].split('/')[-1])
        print(nb_epoch)

    return model, nb_epoch
