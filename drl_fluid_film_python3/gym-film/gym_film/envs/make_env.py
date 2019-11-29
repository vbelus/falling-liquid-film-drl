from gym_film.envs import start_server
import gym
from multiprocessing import Process, Event
import param
import socket
import time
import warnings
warnings.filterwarnings("ignore")


def make_env(method, n_jets=param.n_jets, jets_position=None, n_cpu=param.n_cpu, port=None, host=None, render=False, **kwargs):
    jets_position = param.jets_position if jets_position is None else jets_position
    if method == '1env_njet':
        n_envs = n_cpu
    elif method == '1env_1jet':
        n_envs = n_jets
        # port - default: 12000
        if not port:
            port = 12000
        port = int(port)
        if host is None:
            host = socket.gethostname()

        ####################### Starting the server #######################
        is_set = Event() # we will wait for the server to start
        is_ok = Event()
        Process(target=start_server.start,
                args=(host, port, render, is_set, is_ok)).start()
        is_set.wait()
        if not is_ok.is_set():
            raise Exception("The server could not start, quitting.")
        ###################################################################

    def make_gym_env(**kwargs):
        # first method
        if method == "1env_njet":
            def _env():
                return gym.make('gym_film:film-v0',
                                n_jets=n_jets,
                                jets_position=jets_position,
                                render=render,
                                **kwargs)
        # second method
        elif method == "1env_1jet":
            jet_index = kwargs['jet_index']

            def _env():
                return gym.make('gym_film:film-parallel-v0',
                                port=port,
                                host=host,
                                n_jets=n_jets,
                                jets_position=jets_position[jet_index],
                                render=render,
                                **kwargs)                        
        return _env

    # build the environment
    # it should be a vectorized environment
    envs = []
    for i in range(n_envs):
        _env = make_gym_env(jet_index=i, **kwargs)
        envs.append(_env)
    return envs
