import numpy as np
from scipy.constants import N_A
from pymatgen.util.testing import PymatgenTest
from pymatgen.reactions.reaction_network import ReactionNetwork
from pymatgen.core import Molecule
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.reactions.reaction_propagator_new import ReactionPropagator



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

        self.mol_entries = [H2O, H2O_1, H2O_1p, H2, H2_1, H2_1p,
        OH, OH_1, OH_1p, O2, O2_1p, O2_1, H3O, H3O_1, H3O_1p,
        O, O_1, O_1p, H, H_1p, H_1]

        self.reaction_network = ReactionNetwork.from_input_entries(self.mol_entries)
        self.reaction_network.build()
        # Only H2O, H2, O2 present initially
        self.initial_state = {1: self.concentration, 4: self.concentration, 10: self.concentration}
        self.propagator = ReactionPropagator(self.reaction_network, self.initial_state, self.volume)

        self.total_propensity = 0
        for i, reaction in self.reaction_network.reactions.items():
            if all([self.propagator.state.get(r.entry_id, 0) > 0 for r in reaction.reactants]):
                self.total_propensity += self.propagator.get_propensity(reaction, reverse=False)
            if all([self.propagator.state.get(r.entry_id, 0) > 0 for r in reaction.products]):
                self.total_propensity += self.get_propensity(reaction, reverse=True)

    def test_get_propensity(self):
        ### choose a single molecular reaction with H2O as a reactant
        reaction = self.reaction_network.reactions[0]
        desired_propensity = self.num_mols * reaction.rate_constant()
        actual_propensity = self.propagator.get_propensity(reaction, reverse = False)
        self.assertAlmostEqual(actual_propensity, desired_propensity)
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
            reaction_dict[forward_reaction]["probability"] = self.propagator.get_propensity(reaction, reverse = 0) / self.total_propensity
            reaction_dict[reverse_reaction]["probability"] = self.propagator.get_propensity(reaction, reverse = 1) / self.total_propensity
        for sample in range(num_samples):
            reaction_chosen = self.propagator.reaction_choice()
            reactions_dict[reaction_chosen]['count'] += 1
        expected_frequency = dict()
        actual_frequency = dict()
        for reaction in reaction_dict:
            expected_frequency[reaction] = reactions_dict[reaction]["probability"]
            actual_frequency = reaction_dict[reaction]["count"] / num_samples
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




