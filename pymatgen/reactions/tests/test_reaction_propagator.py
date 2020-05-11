import numpy as np
import random
from scipy.constants import N_A
from pymatgen.util.testing import PymatgenTest
from pymatgen.reactions.reaction_network import ReactionNetwork
from pymatgen.core import Molecule
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.reactions.reaction_propagator_new import ReactionPropagator
import unittest
import copy



__author__ = "Ronald Kam, Evan Spotte-Smith"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"


class TestReactionPropagator(PymatgenTest):
    def setUp(self):
        """ Create an initial state and reaction network, based on H2O molecule.
        Species include H2, H2O, H, O, O2, OH, H3O
        """
        self.volume = 10**-24 ## m^3
        ## 10 molecules each of H2O, H2, O2
        self.num_mols = 100
        self.concentration = self.num_mols / N_A / self.volume / 1000 ## mol/L
        print(self.concentration)

        ## Make molecule objects

        H2O_mol = Molecule.from_file("H2O.xyz")
        H2O_mol1 = copy.deepcopy(H2O_mol)
        H2O_mol_1 = copy.deepcopy(H2O_mol)
        H2O_mol1.set_charge_and_spin(charge = 1)
        H2O_mol_1.set_charge_and_spin(charge = -1)

        H2_mol = Molecule.from_file("H2.xyz")
        H2_mol1 = copy.deepcopy(H2_mol)
        H2_mol_1 = copy.deepcopy(H2_mol)
        H2_mol1.set_charge_and_spin(charge = 1)
        H2_mol_1.set_charge_and_spin(charge = -1)

        O2_mol = Molecule.from_file("O2.xyz")
        O2_mol1 = copy.deepcopy(O2_mol)
        O2_mol_1 = copy.deepcopy(O2_mol)
        O2_mol1.set_charge_and_spin(charge = 1)
        O2_mol_1.set_charge_and_spin(charge = -1)

        OH_mol = Molecule.from_file("OH.xyz")
        OH_mol1 = copy.deepcopy(OH_mol)
        OH_mol_1 = copy.deepcopy(OH_mol)
        OH_mol1.set_charge_and_spin(charge = 1)
        OH_mol_1.set_charge_and_spin(charge = -1)

        H_mol = Molecule.from_file("H.xyz")
        H_mol1 = copy.deepcopy(H_mol)
        H_mol_1 = copy.deepcopy(H_mol)
        H_mol1.set_charge_and_spin(charge = 1)
        H_mol_1.set_charge_and_spin(charge = -1)

        O_mol = Molecule.from_file("O.xyz")
        O_mol1 = copy.deepcopy(O_mol)
        O_mol_1 = copy.deepcopy(O_mol)
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
        OH_1p = MoleculeEntry(OH_mol1, energy = -75.2707068199185, correction = 0, enthalpy = 6.469, entropy = 41.518, parameters= None, entry_id= 9, attribute= None)
        ## O2 10-12
        O2 = MoleculeEntry(O2_mol, energy = -150.291045922131, correction = 0, enthalpy = 4.821, entropy = 46.76, parameters= None, entry_id= 10, attribute= None)
        O2_1p = MoleculeEntry(O2_mol1, energy = -149.995474036502, correction = 0, enthalpy = 5.435, entropy = 46.428, parameters= None, entry_id= 11, attribute= None)
        O2_1 = MoleculeEntry(O2_mol_1, energy = -150.454499528454, correction = 0, enthalpy = 4.198, entropy = 47.192, parameters= None, entry_id= 12, attribute= None)

        ## O 13-15
        O =  MoleculeEntry(O_mol, energy = -74.9760564004, correction = 0, enthalpy = 1.481, entropy = 34.254, parameters= None, entry_id= 13, attribute= None)
        O_1 = MoleculeEntry(O_mol_1, energy = -75.2301047938, correction = 0, enthalpy = 1.481, entropy = 34.254, parameters= None, entry_id= 14, attribute= None)
        O_1p = MoleculeEntry(O_mol1, energy = -74.5266804995, correction = 0, enthalpy = 1.481, entropy = 34.254, parameters= None, entry_id= 15, attribute= None)
        ## H 15-18
        H = MoleculeEntry(H_mol, energy = -0.5004488848, correction = 0, enthalpy = 1.481, entropy = 26.014, parameters= None, entry_id= 16, attribute= None)
        H_1p = MoleculeEntry(H_mol1, energy = -0.2027210483, correction = 0, enthalpy = 1.481, entropy = 26.066, parameters= None, entry_id= 17, attribute= None)
        H_1 = MoleculeEntry(H_mol_1, energy = -0.6430639079, correction = 0, enthalpy = 1.481, entropy = 26.014, parameters= None, entry_id= 18, attribute= None)

        self.mol_entries = [H2O, H2O_1, H2O_1p, H2, H2_1, H2_1p,
        OH, OH_1, OH_1p, O2, O2_1p, O2_1,
        O, O_1, O_1p, H, H_1p, H_1]

        self.reaction_network = ReactionNetwork.from_input_entries(self.mol_entries, electron_free_energy=-2.15)
        self.reaction_network.build()
        print([e.entry_id for e in self.reaction_network.entries_list])
        #print("Total number of reactions is " + str(len(self.reaction_network.reactions)) + " and they are: ")

        # Only H2O, H2, O2 present initially
        self.initial_state = {1: self.concentration, 4: self.concentration, 10: self.concentration}
        self.propagator = ReactionPropagator(self.reaction_network, self.initial_state, self.volume)

        self.total_propensity = 0
        self.propensity_list = list()
        for reaction in self.reaction_network.reactions:
            if all([self.propagator.state.get(r.entry_id, 0) > 0 for r in reaction.reactants]):
                self.total_propensity += self.propagator.get_propensity(reaction, reverse=False)
                self.propensity_list.append(self.propagator.get_propensity(reaction, reverse = False))
            if all([self.propagator.state.get(r.entry_id, 0) > 0 for r in reaction.products]):
                self.total_propensity += self.propagator.get_propensity(reaction, reverse=True)
                self.propensity_list.append(self.propagator.get_propensity(reaction, reverse=True))
            print("RCT", reaction.reactant_ids, "PRO", reaction.product_ids)
            print(reaction.free_energy())
            print(reaction.rate_constant())
            print(self.propagator.get_propensity(reaction, reverse=False))
            print(self.propagator.get_propensity(reaction, reverse=True))

        print("Total Propensity is: ", self.total_propensity)
        print("Number of reactions is: ", len(self.reaction_network.reactions))

    def tearDown(self) -> None:
        del self.volume
        del self.num_mols
        del self.concentration
        del self.mol_entries
        del self.reaction_network
        del self.propagator
        del self.initial_state
        del self.total_propensity
        del self.propensity_list


    # def test_get_propensity(self):
    #     ## choose a single molecular reaction with H2O as a reactant: choose intermolecular H2 --> H+ + H-
    #     for reaction in self.reaction_network.reactions:
    #         if ([r.entry_id for r in reaction.reactants] == [4]) and ([p.entry_id for p in reaction.products] == [18, 17]):
    #             chosen_reaction = reaction
    #     reaction = self.reaction_network.reactions[0]
    #     desired_propensity = self.num_mols * reaction.rate_constant()["k_A"]
    #     actual_propensity = self.propagator.get_propensity(reaction, reverse = False)
    #     self.assertAlmostEqual(actual_propensity, desired_propensity, places = 0, msg = "Propensity is not expected")
    #
    # def test_update_state(self):
    #     for reaction in self.reaction_network.reactions:
    #         if ([r.entry_id for r in reaction.reactants] == [4]) and ([p.entry_id for p in reaction.products] == [18, 17]):
    #             chosen_reaction = reaction
    #     desired_state = copy.deepcopy(self.propagator._state)
    #     desired_state[4] = 99
    #     desired_state[18] = 1
    #     desired_state[17] = 1
    #     actual_state = self.propagator.update_state(chosen_reaction, reverse = False)
    #     print("Actual State:")
    #     print(actual_state)
    #     print("Desired:")
    #     print(desired_state)
    #     self.assertDictEqual(actual_state, desired_state, msg = "State update is not consistent with chosen reaction.")
    #
    #
    # def test_reaction_choice(self):
    #     "Choose reaction from initial state n times, compare frequency of each reaction to the probability of being chosen, based on reaction propensities at initial state."
    #     num_samples = 100000
    #     reactions_dict = dict()
    #     ## Obtain propensity of each reaction, initialize reaction count, and probability of reaction
    #     for reaction in self.reaction_network.reactions:
    #         forward_reaction = (reaction, 0)
    #         reverse_reaction = (reaction, 1)
    #         reactions_dict[forward_reaction] = dict()
    #         reactions_dict[reverse_reaction] = dict()
    #         reactions_dict[forward_reaction]["count"] = 0
    #         reactions_dict[reverse_reaction]["count"] = 0
    #         reactions_dict[forward_reaction]["probability"] = self.propagator.get_propensity(reaction, reverse = 0) / self.total_propensity
    #         reactions_dict[reverse_reaction]["probability"] = self.propagator.get_propensity(reaction, reverse = 1) / self.total_propensity
    #     for sample in range(num_samples):
    #         reaction_chosen = self.propagator.reaction_choice()
    #         reactions_dict[reaction_chosen]["count"] += 1
    #     expected_frequency = np.array([])
    #     actual_frequency = np.array([])
    #     for reaction in reactions_dict:
    #         expected_frequency = np.append(expected_frequency, reactions_dict[reaction]["probability"])
    #         actual_frequency = np.append(actual_frequency, reactions_dict[reaction]["count"] / num_samples)
    #
    #     print("Expected frequencies of reaction choice")
    #     print(expected_frequency)
    #     print("Actual frequencies of reaction choice")
    #     print(actual_frequency)
    #     self.assertArrayAlmostEqual(expected_frequency, actual_frequency, decimal = 2, err_msg = "Reaction choice frequency is not consistent with initial state")

    def test_simulate(self):
        t_end = 10 ** (-9)
        simulation_data = self.propagator.simulate(t_end)
        time_record = simulation_data["times"]
        self.propagator.plot_trajectory(simulation_data, "Simulation Results")
        self.assertAlmostEqual(time_record[-1], t_end, 10)
        expected_tau = 1/self.total_propensity
        tau_list = np.diff(time_record)
        self.assertAlmostEqual(np.average(tau_list), expected_tau)

if __name__ == "__main__":
    unittest.main()


