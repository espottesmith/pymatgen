import numpy as np
import random
from scipy.constants import N_A
from pymatgen.util.testing import PymatgenTest
from pymatgen.reactions.reaction_network import ReactionNetwork
from pymatgen.core import Molecule
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.reactions.reaction_propagator_new import ReactionPropagator
import unittest



__author__ = "Ronald Kam, Evan Spotte-Smith"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"


class TestReactionPropagator(PymatgenTest):
    @classmethod
    def setUpClass(self):
        """ Create an initial state and reaction network, based on H2O molecule.
        Species include H2, H2O, H, O, O2, OH, H3O

        """
        self.volume = 10**-24 ## m^3
        ## 10 molecules each of H2O, H2, O2
        self.num_mols = 10
        self.concentration = 10 / N_A /self.volume / 1000 ## mol/L

        ## Make molecule objects

        H2O_mol = Molecule.from_file("H2O.xyz")
        H2O_mol1 = H2O_mol
        H2O_mol_1 = H2O_mol
        H2O_mol1.set_charge_and_spin(charge = 1)
        H2O_mol_1.set_charge_and_spin(charge = -1)
        H2_mol = Molecule.from_file("H2.xyz")
        H2_mol1 = H2_mol
        H2_mol_1 = H2_mol
        H2_mol1.set_charge_and_spin(charge = 1)
        H2_mol_1.set_charge_and_spin(charge = -1)

        O2_mol = Molecule.from_file("O2.xyz")
        O2_mol1 = O2_mol
        O2_mol_1 = O2_mol
        O2_mol1.set_charge_and_spin(charge = 1)
        O2_mol_1.set_charge_and_spin(charge = -1)

        OH_mol = Molecule.from_file("OH.xyz")
        OH_mol1 = OH_mol
        OH_mol_1 = OH_mol
        OH_mol1.set_charge_and_spin(charge = 1)
        OH_mol_1.set_charge_and_spin(charge = -1)

        H3O_mol = Molecule.from_file("H3O.xyz")
        H3O_mol1 = H3O_mol
        H3O_mol_1 = H3O_mol
        H3O_mol1.set_charge_and_spin(charge = 1)
        H3O_mol_1.set_charge_and_spin(charge = -1)

        H_mol = Molecule.from_file("H.xyz")
        H_mol1 = H_mol
        H_mol_1 = H_mol
        H_mol1.set_charge_and_spin(charge = 1)
        H_mol_1.set_charge_and_spin(charge = -1)

        O_mol = Molecule.from_file("O.xyz")
        O_mol1 = O_mol
        O_mol_1 = O_mol
        O_mol1.set_charge_and_spin(charge = 1)
        O_mol_1.set_charge_and_spin(charge = -1)

## Making molecule entries
        ## H2O 1-3
        H2O = MoleculeEntry(H2O_mol, energy = -76.4447861695239, correction = 0, enthalpy = 15.702, entropy = 46.474, parameters= None, entry_id= 1, attribute= None)
        H2O_1 = MoleculeEntry(H2O_mol_1, energy = -76.4634569330715, correction = 0, enthalpy = 13.298, entropy = 46.601, parameters= None, entry_id= 2, attribute= None)
        H2O_1p = MoleculeEntry(H2O_mol1, energy = -76.0924662469782, correction = 0, enthalpy = 13.697, entropy = 46.765, parameters= None, entry_id= 3, attribute= None)
        ## H2 4-6
        H2 = MoleculeEntry(H2_mol, energy = -1.17275734244991, correction = 0, enthalpy = 8.685, entropy = 31.141, parameters= None, entry_id= 4, attribute= None)
        H2_1 = MoleculeEntry(H2_mol_1, energy = -1.16232420718418, correction = 0, enthalpy = 3.56, entropy = 33.346, parameters= None, entry_id= 5, attribute= None)
        H2_1p = MoleculeEntry(H2_mol1, energy = -0.781383960574136, correction = 0, enthalpy = 5.773, entropy = 32.507, parameters= None, entry_id= 6, attribute= None)

        ## OH 7-9
        OH = MoleculeEntry(OH_mol, energy = -75.7471080255785, correction = 0, enthalpy = 7.659, entropy = 41.21, parameters= None, entry_id= 7, attribute= None)
        OH_1 = MoleculeEntry(OH_mol_1, energy = -75.909589774742, correction = 0, enthalpy = 7.877, entropy = 41.145, parameters= None, entry_id= 8, attribute= None)
        OH_1p = MoleculeEntry(OH_mol_1, energy = -75.2707068199185, correction = 0, enthalpy = 6.469, entropy = 41.518, parameters= None, entry_id= 9, attribute= None)
        ## O2 10-12
        O2 = MoleculeEntry(O2_mol, energy = -150.291045922131, correction = 0, enthalpy = 4.821, entropy = 46.76, parameters= None, entry_id= 10, attribute= None)
        O2_1p = MoleculeEntry(O2_mol1, energy = -149.995474036502, correction = 0, enthalpy = 5.435, entropy = 46.428, parameters= None, entry_id= 11, attribute= None)
        O2_1 = MoleculeEntry(O2_mol_1, energy = -150.454499528454, correction = 0, enthalpy = 4.198, entropy = 47.192, parameters= None, entry_id= 12, attribute= None)
        ## H3O 13-15
        H3O = MoleculeEntry(H3O_mol, energy = -76.9068557089757, correction = 0, enthalpy = 14.809, entropy = 48.818, parameters= None, entry_id= 13, attribute= None)
        H3O_1 = MoleculeEntry(H3O_mol_1, energy = -76.9648792962602, correction = 0, enthalpy = 14.021, entropy = 49.233, parameters= None, entry_id= 14, attribute= None)
        H3O_1p = MoleculeEntry(H3O_mol1, energy = -76.9068557089757, correction = 0, enthalpy = 23.612, entropy = 48.366, parameters= None, entry_id= 15, attribute= None)
        ## O 16-18
        O =  MoleculeEntry(O_mol, energy = -74.9760564004, correction = 0, enthalpy = 1.481, entropy = 34.254, parameters= None, entry_id= 16, attribute= None)
        O_1 = MoleculeEntry(O_mol_1, energy = -75.2301047938, correction = 0, enthalpy = 1.481, entropy = 34.254, parameters= None, entry_id= 17, attribute= None)
        O_1p = MoleculeEntry(O_mol1, energy = -74.5266804995, correction = 0, enthalpy = 1.481, entropy = 34.254, parameters= None, entry_id= 18, attribute= None)
        ## H 19-21
        H = MoleculeEntry(H_mol, energy = -0.5004488848, correction = 0, enthalpy = 1.481, entropy = 26.014, parameters= None, entry_id= 19, attribute= None)
        H_1p = MoleculeEntry(H_mol1, energy = -0.2027210483, correction = 0, enthalpy = 1.481, entropy = 26.066, parameters= None, entry_id= 20, attribute= None)
        H_1 = MoleculeEntry(H_mol_1, energy = -0.6430639079, correction = 0, enthalpy = 1.481, entropy = 26.014, parameters= None, entry_id= 21, attribute= None)

        self.mol_entries = [H2O, H2O_1, H2O_1p, H2, H2_1, H2_1p,
        OH, OH_1, OH_1p, O2, O2_1p, O2_1, H3O, H3O_1, H3O_1p,
        O, O_1, O_1p, H, H_1p, H_1]

        self.reaction_network = ReactionNetwork.from_input_entries(self.mol_entries, electron_free_energy=-2.15)
        self.reaction_network.build()
        print("Total number of reactions is " + str(len(self.reaction_network.reactions)))
        # Only H2O, H2, O2 present initially
        self.initial_state = {1: self.concentration, 4: self.concentration, 10: self.concentration}
        self.propagator = ReactionPropagator(self.reaction_network, self.initial_state, self.volume)

        self.total_propensity = 0
        for reaction in self.reaction_network.reactions:
            if all([self.propagator.state.get(r.entry_id, 0) > 0 for r in reaction.reactants]):
                self.total_propensity += self.propagator.get_propensity(reaction, reverse=False)
            if all([self.propagator.state.get(r.entry_id, 0) > 0 for r in reaction.products]):
                self.total_propensity += self.get_propensity(reaction, reverse=True)
        print('Total propensity is ' + str(self.total_propensity))
    def test_get_propensity(self):
        ### choose a single molecular reaction with H2O as a reactant
        reaction = self.reaction_network.reactions[0]
        desired_propensity = self.num_mols * reaction.rate_constant()
        actual_propensity = self.propagator.get_propensity(reaction, reverse = False)
        self.assertAlmostEqual(actual_propensity, desired_propensity, decimal = 7, err_msg = "Propensity is not expected")
    def test_update_state(self):
        reaction = self.reaction_network.reactions[0]
        actual_state = self.propagator.update_state(reaction, reverse = False)
        desired_state = self.initial_state
        desired_state[1] -= 1/ N_A /self.volume / 1000 # remove one H2O
        self.assertDictsAlmostEqual(actual_state, desired_state, decimal = 7, err_msg = "State update is not consistent with chosen reaction.")

    def test_reaction_choice(self):
        "Choose reaction from initial state n times, compare frequency of each reaction to the probability of being chosen, based on reaction propensities at initial state."
        num_samples = 100000
        reactions_dict = dict()
        ## Obtain propensity of each reaction, initialize reaction count, and probability of reaction
        for reaction in self.reaction_network.reactions:
            forward_reaction = (reaction, 0)
            reverse_reaction = (reaction, 1)
            reactions_dict[forward_reaction] = dict()
            reactions_dict[reverse_reaction] = dict()
            reactions_dict[forward_reaction]["count"] = 0
            reactions_dict[reverse_reaction]["count"] = 0
            reactions_dict[forward_reaction]["probability"] = self.propagator.get_propensity(reaction, reverse = 0) / self.total_propensity
            reactions_dict[reverse_reaction]["probability"] = self.propagator.get_propensity(reaction, reverse = 1) / self.total_propensity
        for sample in range(num_samples):
            reaction_chosen = self.propagator.reaction_choice()
            reactions_dict[reaction_chosen]['count'] += 1
        expected_frequency = dict()
        actual_frequency = dict()
        for reaction in reactions_dict:
            expected_frequency[reaction] = reactions_dict[reaction]["probability"]
            actual_frequency = reactions_dict[reaction]["count"] / num_samples
        self.assertDictsAlmostEqual(expected_frequency, actual_frequency, decimal = 4, err_msg = "Reaction choice frequency is not consistent with initial state")

    def test_time_step(self):
         num_samples = 1000
         time_steps = list()
         for sample in range(num_samples):
            tau = -np.log(random.random()) / self.total_propensity
            time_steps.append(tau)
         average_tau = np.average(time_steps)
         expected_tau = 1 / self.total_propensity
         print("Average initial time step is " + average_tau + ", and expected value is " + expected_tau)
         self.assertAlmostEqual(average_tau, expected_tau, decimal = 7)

if __name__ == "__main__":
    unittest.main()


