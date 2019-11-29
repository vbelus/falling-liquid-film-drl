import numpy as np
from gym_film.envs.simulation_solver import cpp
import param
import os
import pickle
import itertools


class SimulationNoPertubationJet():
    def __init__(self, jets_position=None, starting_system_state=None, dtype='double'):
        self.jets_position = param.jets_position if jets_position is None else jets_position
        self.starting_system_state = np.ones(
            (2, param.NUM)) if starting_system_state is None else starting_system_state
        self.C = cpp.Simulation_cpp(self.jets_position.astype('int32'),
                                    param.L,
                                    param.NUM,
                                    param.dx,
                                    param.dt,
                                    param.delta,
                                    param.noise_mag,
                                    param.n_step,
                                    param.nb_timestep_per_simulation,
                                    param.JET_WIDTH//2)
        self.C.set_h(self.starting_system_state[0])
        self.C.set_q(self.starting_system_state[1])
        self.reset()

    def next_step(self, jet_power):
        jet_power = np.array(jet_power)
        self.C.set_jet_power(jet_power.astype('double'))
        self.C.next_step()
        self.h = self.C.get_h()
        self.q = self.C.get_q()
        self.t = self.C.get_time()
        print(self.t)

    def _no_jet_next_step(self):
        self.next_step(np.zeros(param.n_jets))

    def get_system_state(self):
        return np.array([self.h, self.q])

    def get_time(self):
        return self.t

    def _get_starting_state(self, starting_system_state,
                            initial_waiting_n_step=param.initial_waiting_n_step,
                            pre_simulation=True):
        if pre_simulation:
            filename = os.path.join(os.path.dirname(
                __file__), 'starting_states/',
                str(param.initial_waiting_time)+"_"+str(param.L) + '.pkl')
            if (param.new_params or not os.path.exists(filename)):
                for _ in itertools.repeat(None, param.initial_waiting_n_step):
                    self._no_jet_next_step()
                starting_state_file = open(filename, 'wb')
                starting_system_state = self.get_system_state()
                pickle.dump(starting_system_state, starting_state_file)
            else:
                starting_state_file = open(filename, 'rb')
                starting_system_state = pickle.load(starting_state_file)
        return starting_system_state

    def reset(self):
        starting_system_state = self._get_starting_state(
            self.starting_system_state)
        self.h = starting_system_state[0]
        self.q = starting_system_state[1]
        self.t = 0.

        self.C.set_h(self.h)
        self.C.set_q(self.q)


class Simulation(SimulationNoPertubationJet):
    def __init__(self, perturbation_jets_position=None):
        perturbation_jets_position = param.perturbation_jets_position if perturbation_jets_position is None else perturbation_jets_position

        # we process the array of perturbation jets positions
        perturbation_jets_position = np.array(perturbation_jets_position)
        perturbation_jets_position = np.array(
            perturbation_jets_position/param.dx, dtype="int")
        # we concatenate the two arrays
        jets_position = np.concatenate((
            perturbation_jets_position, param.jets_position))
        self.n_perturbation_jets = len(perturbation_jets_position)
        self.n_jets = len(jets_position)
        super(Simulation, self).__init__(jets_position=jets_position)

    def next_step(self, jet_power):
        jet_power = np.array(jet_power)
        # we set random values for the perturbation jets, in the good interval
        perturbation_jets_power = param.perturbation_jets_power * \
            (1-2*np.random.random_sample((self.n_perturbation_jets,)))
        # we concatenate the two arrays
        jet_power = np.concatenate((perturbation_jets_power, jet_power))

        self.C.set_jet_power(jet_power.astype('double'))
        self.C.next_step()
        self.h = self.C.get_h()
        self.q = self.C.get_q()
        self.t = self.C.get_time()

    def _no_jet_next_step(self):
        self.next_step(np.zeros(self.n_jets))
