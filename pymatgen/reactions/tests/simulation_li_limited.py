import numpy as np
from pymatgen.core import Molecule
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.reactions.reaction_propagator_new import ReactionPropagator
from monty.serialization import dumpfn, loadfn
from pymatgen.reactions.reaction_network import ReactionNetwork
import time
import matplotlib.pyplot as plt
import pickle



__author__ = "Ronald Kam, Evan Spotte-Smith"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"

class Simulation_Li_Limited:
    def __init__(self, file_name, li_conc = 1.0, ec_conc = 3.5706, emc_conc = 7.0555, volume = 10**-24, t_end = 1):
        """ Create an initial state and reaction network, in a Li system of ~ 3200 molecules.
        Typical electrolyte composition is 1M LiPF6, 3:7 wt% EC:EMC

        Args:
        li_conc (float): Li concentration
        ec_conc (float): Ethylene carbonate concentration
        emc_conc (float): Ethyl methyl carbonate
        volume (float): Volume in Liters (default = 1 nm^3 = 1 * 10^-24 L)
        t_end (float): end time of simulation
        """
        # Set up initial conditions, use baseline Li-ion electrolyte solution
        self.volume = volume ## m^3
        self.li_conc = li_conc # mol/L
        self.file_name = file_name
        self.ec_conc = ec_conc
        self.emc_conc = emc_conc
        self.t_end = t_end
        # Impurities
        self.h2o_conc = 1.665*10**-4 # 1-5 ppm
        #self.hf_conc = 2.70*10**-3 # 30-60 ppm

        # ref_ec = Molecule.from_file("ref_ec.xyz")
        # ref_ec = MoleculeGraph.with_local_env_strategy(ref_ec, OpenBabelNN())
        # ref_emc = Molecule.from_file("ref_emc.xyz")
        # ref_emc = MoleculeGraph.with_local_env_strategy(ref_emc, OpenBabelNN())
        # ref_h2o = Molecule.from_file("ref_h2o.xyz")
        # ref_h2o = MoleculeGraph.with_local_env_strategy(ref_h2o, OpenBabelNN())

        # Put entries in a list to make ReactionNetwork
        # self.entries = loadfn("mol_entries_limited_two.json")
        #
        # for ii, entry in enumerate(self.entries):
        #     entry.entry_id = ii
        # pickle_out = open("pickle_mol_entries_limited_two_IDs", "wb")
        # pickle.dump(self.entries, pickle_out)
        # pickle_out.close()

        # time_start = time.time()
        # self.reaction_network = ReactionNetwork.from_input_entries(self.entries, electron_free_energy = -2.15)
        # self.reaction_network.build()
        # time_end = time.time()
        #print("Time to generate rxn network: ", time_end-time_start)
        # pickle_out = open("pickle_rxnnetwork_Li-limited", "wb")
        # pickle.dump(self.reaction_network, pickle_out)
        # pickle_out.close()

        pickle_in = open("pickle_rxnnetwork_Li-limited", "rb")
        self.reaction_network = pickle.load(pickle_in)

        li_id = 2335
        ec_id = 2606
        emc_id = 1877
        h2o_id = 3306

        # start_time = time.time()
        # self.reaction_network = loadfn("rxn_network_mol_entries_limited_two.json")
        # end_time = time.time()
        #print("Time to load reaction network: ", end_time - start_time)

        self.initial_state = {li_id: self.li_conc, ec_id: self.ec_conc, emc_id: self.emc_conc, h2o_id: self.h2o_conc}
        self.propagator = ReactionPropagator(self.reaction_network, self.initial_state, self.volume)
        #
        prd_to_change = [1357, 5061] # want to eliminate CH3, C2H5 formation
        for id in prd_to_change:
            self.propagator.alter_rxn_by_product(id, 1/10000)

        print("Initial state is: ", self.propagator.state)
        time_start = time.time()
        self.simulation_data = self.propagator.simulate(self.t_end)
        time_end = time.time()
        print("Total simulation time is: ", time_end - time_start)
        self.propagator.plot_trajectory(self.simulation_data,"Simulation Results", self.file_name)
        print("Final state is: ", self.propagator.state)
        self.rxn_analysis = self.propagator.reaction_analysis()
        print("Reaction Analysis")
        for analysis_key in self.rxn_analysis:
            print(self.rxn_analysis[analysis_key])

        #pickle_out = open("pickle_simdata_" + self.file_name, "wb")
        #pickle.dump(self.simulation_data, pickle_out)
        #pickle_out.close()

    def time_analysis(self):
        time_dict = dict()
        time_dict["t_avg"] = np.average(self.simulation_data["times"])
        time_dict["t_std"] = np.std(self.simulation_data["times"])
        time_dict["steps"] = len(self.simulation_data["times"])
        time_dict["total_t"] = self.simulation_data["times"][-1]
        return time_dict

li_conc = 1.0
# 3:7 EC:EMC
ec_conc = 3.57
emc_conc = 7.0555
volume = 10**-24
t_end = 10**-10
this_simulation = Simulation_Li_Limited("li_limited_t_1e-10_ea_10000_noCH3_C2H5", li_conc, ec_conc, emc_conc, volume, t_end)
time_data = this_simulation.time_analysis()
print("Time analysis")
print(time_data)

# times = [10**-13, 10**-12, 10**-11, 10**-10, 10**-9]
# runtime_data = dict()
# runtime_data["runtime"] = list()
# runtime_data["t_avg"] = list()
# runtime_data["t_std"] = list()
# runtime_data["steps"] = list()
#
# for t_end in times:
#     this_filename = "Simulation_Run_" + str(t_end)
#     this_simulation = Simulation_Li_Limited(this_filename, li_conc, ec_conc, emc_conc, volume, t_end)
#     time_data = this_simulation.time_analysis()
#     runtime_data["t_avg"].append(time_data["t_avg"])
#     runtime_data["t_std"].append(time_data["t_std"])
#     runtime_data["steps"].append(time_data["steps"])
#     runtime_data["runtime"].append(this_simulation.runtime)
#
# plt.figure()
# plt.subplot(411)
# plt.plot(times, runtime_data["t_avg"])
# plt.title("Average Time Steps")
# plt.ylabel("Time (s)")
# plt.xlabel("t_end (s)")
#
# plt.subplot(412)
# plt.plot(times, runtime_data["t_std"])
# plt.title("Std Dev Time Steps")
# plt.ylabel("Time (s)")
# plt.xlabel("t_end (s)")
#
# plt.subplot(413)
# plt.plot(times, runtime_data["steps"])
# plt.title("Number of Time Steps")
# plt.ylabel("Time steps")
# plt.xlabel("t_end (s)")
#
# plt.subplot(414)
# plt.plot(times, runtime_data["runtime"])
# plt.title("Simulation Runtime Analysis")
# plt.ylabel("Runtime (s)")
# plt.xlabel("t_end (s)")
#
# plt.savefig("Simulation_Time_Analysis_t-9")
