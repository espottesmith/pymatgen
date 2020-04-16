import numpy as np
import unittest
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.util.testing import PymatgenTest
from pymatgen.reactions import ReactionNetwork
from pymatgen.core import Molecule
import os

__author__ = "Ronald Kam, Evan Spotte-Smith"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"

N = 6.0221409e+23

class TestReactionPropagator(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        """ Create an initial state and reaction network, based on H2O molecule.
        Species include H2, H2O, H, O, O2, OH, H3O

        """
        cls.volume = 10**-24 ## m^3
        ## 1 molecule each of H2O, H2, O2
        cls.concentration = 1 / N /cls.volume / 1000 ## mol/L

        ## Make molecule objects

        H2O_mol = Molecule.from_file("H2O.xyz")
        H2O_mol1 = H2O_mol.set_charge_and_spin(1)
        H2O_mol_1 = H2O_mol.set_charge_and_spin(-1)

        H2_mol = Molecule.from_file("H2.xyz")
        H2_mol1 = H2_mol.set_charge_and_spin(1)
        H2_mol_1 = H2_mol.set_charge_and_spin(-1)

        O2_mol = Molecule.from_file("O2.xyz")
        O2_mol1 = O2_mol.set_charge_and_spin(1)
        O2_mol_1 = O2_mol.set_charge_and_spin(-1)

        OH_mol = Molecule.from_file("OH.xyz")
        OH_mol1 = OH_mol.set_charge_and_spin(1)
        OH_mol_1 = OH_mol.set_charge_and_spin(-1)

        H3O_mol = Molecule.from_file("H3O.xyz")
        H3O_mol1 = H3O_mol.set_charge_and_spin(1)
        H3O_mol_1 = H3O_mol.set_charge_and_spin(-1)

        H_mol = Molecule.from_file("H.xyz")
        H_mol1 = H_mol.set_charge_and_spin(1)
        H_mol_1 = H_mol.set_charge_and_spin(-1)

        O_mol = Molecule.from_file("O.xyz")
        O_mol1 = O_mol.set_charge_and_spin(1)
        O_mol_1 = O_mol.set_charge_and_spin(-1)

## Making molecule entries
        ## H2O 1-3
        H2O = MoleculeEntry(H2O_mol, -76.4447861695239, 0, 15.702, 46.474, None, 1, None)
        H2O_1 = MoleculeEntry(H2O_mol_1, -76.4634569330715, 0, 13.298, 46.601, None, 2, None)
        H2O_1p = MoleculeEntry(H2O_mol1, -76.0924662469782, 0, 13.697, 46.765, None, 3, None)
        ## H2 4-6
        H2 = MoleculeEntry(H2_mol, -1.17275734244991, 0, 8.685, 31.141, None, 4, None)
        H2_1 = MoleculeEntry(H2_mol_1, -1.16232420718418, 0, 3.56, 33.346, None, 5, None)
        H2_1p = MoleculeEntry(H2_mol1, -0.781383960574136, 0, 5.773, 32.507, None, 6, None)

        ## OH 7-9
        OH = MoleculeEntry(OH_mol, -75.7471080255785, 0, 7.659, 41.21, None, 7, None)
        OH_1 = MoleculeEntry(OH_mol_1, -75.909589774742, 0, 7.877, 41.145, None, 8, None)
        OH_1p = MoleculeEntry(OH_mol_1, -75.2707068199185, 0, 6.469, 41.518, None, 9, None)
        ## O2 10-12
        O2 = MoleculeEntry(O2_mol, -150.291045922131, 0, 4.821, 46.76, None, 10, None)
        O2_1p = MoleculeEntry(O2_mol1, -149.995474036502, 0, 5.435, 46.428, None, 11, None)
        O2_1 = MoleculeEntry(O2_mol_1, -150.454499528454, 0, 4.198, 47.192, None, 12, None)
        ## H3O 13-15
        H3O = MoleculeEntry(H3O_mol, -76.9068557089757, 0, 14.809,  48.818, None, 13, None)
        H3O_1 = MoleculeEntry(H3O_mol_1, -76.9648792962602, 0, 14.021, 49.233, None, 14, None)
        H3O_1p = MoleculeEntry(H3O_mol1, -76.9068557089757, 0, 23.612, 48.366, None, 15, None)
        ## O 16-18
        O =  MoleculeEntry(O_mol, -74.9760564004, 0, 1.481,  34.254, None, 16, None)
        O_1 = MoleculeEntry(O_mol_1, -75.2301047938, 0, 1.481,  34.254, None, 17, None)
        O_1p = MoleculeEntry(O_mol1, -74.5266804995, 0, 1.481,  34.254, None, 18, None)
        ## H 19-21
        H = MoleculeEntry(H_mol, -0.5004488848, 0, 1.481,  26.014, None, 19, None)
        H_1p = MoleculeEntry(H_mol1, -0.2027210483, 0, 1.481,  26.066, None, 20, None)
        H_1 = MoleculeEntry(H_mol_1, -0.6430639079, 0, 1.481,  26.014, None, 21, None)

        cls.mol_entries = [H2O, H2O_1, H2O_1p, H2, H2_1, H2_1p,
        OH, OH_1, OH_1p, O2, O2_1p, O2_1, H3O, H3O_1, H3O_1p,
        O, O_1, O_1p, H, H_1p, H_1]

        cls.reaction_network = ReactionNetwork.from_input_entries(mol_entries)
        # Only H2O, H2, O2 present initially
        cls.inital_state = {1: cls.concentration, 4: cls.concentration, 10: cls.concentration}
        cls.propagator = ReactionPropagator(cls.reaction_network, cls.initial_state, cls.volume)


    def test_get_propensity(self):
        reaction = self.reaction_network.reactions[0]
        print(reaction.reactants)
