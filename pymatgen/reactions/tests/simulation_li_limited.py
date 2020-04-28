import numpy as np
import random
from scipy.constants import N_A
from pymatgen.util.testing import PymatgenTest
from pymatgen.reactions.reaction_network import ReactionNetwork
from pymatgen.core import Molecule
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.reactions.reaction_propagator_new import ReactionPropagator
import monty
import copy

__author__ = "Ronald Kam, Evan Spotte-Smith"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"

class Simulation_Li_Limited:
    def __init__(self, li_conc = 1.2, ec_conc, emc_conc, volume = 10**-24, t_end):
        """ Create an initial state and reaction network, in a Li system of ~ 3200 molecules
        """
        # Set up initial conditions, use baseline Li-ion electrolyte solution
        self.volume = volume ## m^3
        self.li_conc = li_conc # mol/L
        # 3:7 EC:EMC by wt
        # find these values later
        self.ec_conc = ec_conc
        self.emc_conc = emc_conc
        self.t_end = t_end
        self.h2o_conc = 10**-5

        ref_ec = Molecule.from_file("ref_ec.xyz")
        ref_ec = MoleculeGraph.with_local_env_strategy(ref_ec, OpenBabelNN())
        ref_emc = Molecule.from_file("ref_emc.xyz")
        ref_emc = MoleculeGraph.with_local_env_strategy(ref_emc, OpenBabelNN())
        ref_h2o = Molecule.from_file("ref_h2o.xyz")
        ref_h2o = MoleculeGraph.with_local_env_strategy(ref_h2o, OpenBabelNN())

        self.mol_entries_limited = list()
        ## Put entries in a list to make ReactionNetwork
        entries = loadfn(os.path.join(test_dir,"mol_entries_limited.json"))
        for ii, entry in enumerate(entries):
            mol = entry["output"]["optimized_molecule"]
            E = float(entry["output"]["final_energy"])
            H = float(entry["output"]["enthalpy"])
            S = float(entry["output"]["entropy"])
            mol_entry = MoleculeEntry(molecule = mol, energy = E, enthalpy = H, entropy = S, entry_id = ii)
            self.mol_entries_limited.append(mol_entry)
            # Just to find the mol ID of H2O, Li+, EMC, EC; can delete later
            if entry.mol_graph.isomorhpic_to(ref_ec):
                print("EC: ", ii)
                ec_id = ii
            elif entry.mol_graph.isomorhpic_to(ref_h2o):
                print("H2O: ", ii)
                h2o_id = ii
            elif entry.mol_graph.isomorhpic_to(ref_emc):
                print("EMC: ", ii)
                emc_id = ii

        self.reaction_network = ReactionNetwork.from_input_entries(self.mol_entries_limited, electron_free_energy = -2.15)
        self.reaction_network.build()

        self.initial_state = {ec_id: self.ec_conc, emc_id: self.emc_conc, h2o_id: self.h2o_conc }

li_conc = 1.2
EC_conc = 100
EMC_conc = 300
volume = 10**-24
t_end = 10**-10
this_simulation = Simulation_Li_Limited()


