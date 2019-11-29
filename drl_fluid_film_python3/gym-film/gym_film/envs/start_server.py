from gym_film.envs.simulation_solver.simulation import Simulation
from gym_film.envs.env_server import FilmEnvServer
import socket


def check_free_port(host, port):
    """Check if a given port is available."""
    sock = socket.socket()
    try:
        sock.bind((host, port))
        sock.close()
        print("host {} on port {} is AVAILABLE".format(host, port))
        return(True)
    except Exception as e:
        print(e)
        print("host {} on port {} is BUSY".format(host, port))
        sock.close()
        return(False)


def start(host, port, render, is_set, is_ok):
    if not check_free_port(host, port):
        is_ok.clear()
        is_set.set()
        raise Exception("The port is not available, quitting.")

    else:
        is_ok.set()
        is_set.set()
        FilmEnvServer(Simulation(), host=host, port=port, render=render)

