# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import random
import numpy as np
import matplotlib.pyplot as plt
import time


__author__ = "Ronald Kam, Evan Spotte-Smith"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"
__credit__ = "Xiaowei Xie"

k_b = 1.38064852e-23
T = 298.15
h = 6.62607015e-34
R = 8.3144598
N = 6.0221409e+23


class ReactionPropagator:
    """
    Class for stochastic kinetic Monte Carlo simulation, with reactions provided
    by the Reactions present in a ReactionNetwork.
    Method is described by Gillespie (1976)

    Args:
        reaction_network (ReactionNetwork)
        initial_state (dict): {Molecule ID (int): concentration (float)}}
        volume (float): Volume in Liters (default = 1 nm^3 = 1 * 10^-24 L)

    """
    def __init__(self, reaction_network, initial_state, volume=1.0*10**-24):
        self.reaction_network = reaction_network
        ## make a dict, assigning an index for each reaction. Each index is a dict containing reaction object and will later add propensity
        self.reactions = dict()
        for id, reaction in enumerate(self.reaction_network.reactions):
            self.reactions[id] = reaction
        self.initial_state_conc = initial_state
        self.volume = volume
        self._state = dict()
        self.initial_state = dict()
        ## State will have number of molecules, instead of concentration
        for molecule_id, concentration in self.initial_state_conc.items():
            num_mols = int(concentration * self.volume * N  *1000)# volume in m^3
            self.initial_state[molecule_id] = num_mols
            self._state[molecule_id] = num_mols
        self.data = {"times": list(),
                     "reactions": list(),
                     "state": dict()}

    @property
    def state(self):
        return self._state

    def get_propensity(self, reaction, reverse):
        """
        Calculate the propensity for a particular reaction, based on the
        number of molecules for the reactants and the reaction rate constant.

        Args:
            reaction (Reaction)
            reverse (bool): If True, give the propensity for the reverse
                reaction. If False, give the propensity for the forwards
                reaction.

        Returns:
            propensity (float)
        """

        rate_constant = reaction.rate_constant()
        if reverse:
            k = rate_constant["k_B"]
            num_reactants = len(reaction.products)
            reactants = reaction.products
        else:
            k = rate_constant["k_A"]
            num_reactants = len(reaction.reactants)
            reactants = reaction.reactants

        num_mols_list = list()
        entry_ids = list() # for testing
        for reactant in reactants:
            reactant_num_mols = self.state.get(reactant.entry_id, 0)
            num_mols_list.append(reactant_num_mols)
            entry_ids.append(reactant.entry_id)
        if num_reactants == 1:
            h_prop = num_mols_list[0]
        elif (num_reactants == 2) and (reactants[0].entry_id == reactants[1].entry_id):
            h_prop = num_mols_list[0] * (num_mols_list[0] - 1) / 2
        elif (num_reactants == 2) and (reactants[0].entry_id != reactants[1].entry_id):
            h_prop = num_mols_list[0] * num_mols_list[1]
        else:
            raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
        propensity = h_prop * k
        return propensity
        # for testing:
        #return [reaction.reaction_type, reaction.reactants, reaction.products, reaction.rate_calculator.alpha , reaction.transition_state, "propensity = " + str(propensity), "free energy from code = " + str(reaction.free_energy()["free_energy_A"]), "calculated free energy ="  + str(-sum([r.free_energy() for r in reaction.reactants]) +  sum([p.free_energy() for p in reaction.products])),
                #"calculated k = " +  str(k_b * 298.15 / h * np.exp(-1 * (-sum([r.free_energy() for r in reaction.reactants]) +  sum([p.free_energy() for p in reaction.products]) ) * 96487 / (R * 298.15))), "k from Rxn class = " +  str(k)  ]

    def update_state(self, reaction, reverse):
        """ Update the system based on the reaction chosen
        Args:
            reaction (Reaction)
            reverse (bool): If True, let the reverse reaction proceed.
                Otherwise, let the forwards reaction proceed.

        Returns:
            None
        """
        if reverse:
            for reactant in reaction.products:
                self._state[reactant.entry_id] -= 1
            for product in reaction.reactants:
                p_id = product.entry_id
                if p_id in self.state:
                    self._state[p_id] += 1
                else:
                    self._state[p_id] = 1
        else:
            for reactant in reaction.reactants:
                self._state[reactant.entry_id] -= 1
            for product in reaction.products:
                p_id = product.entry_id
                if p_id in self.state:
                    self._state[p_id] += 1
                else:
                    self._state[p_id] = 1
        return self._state # for testing

    def simulate(self, t_end):
        """
        Main body code of the KMC simulation. Propagates time and updates species amounts.
        Store reactions, time, and time step for each iteration

        Args:
            t_end: (float) ending time of simulation

        Returns
            final state of molecules
        """
        # If any change have been made to the state, revert them
        self._state = self.initial_state
        t = 0.0
        self.data = {"times": list(),
                     "reactions": list(),
                     "state": dict()}

        for mol_id in self._state.keys():
            self.data["state"][mol_id] = [(0.0, self._state[mol_id])]

        step_counter = 0
        while t < t_end:
            step_counter += 1

            ## Obtain reaction propensities, on which the probability distributions of
            ## time and reaction choice depends.
            time_start = time.time()
            total_propensity = 0
            for i, reaction in self.reactions.items():
                if all([self._state.get(r.entry_id, 0) > 0 for r in reaction.reactants]):
                    total_propensity += self.get_propensity(reaction, reverse=False)
                if all([self._state.get(r.entry_id, 0) > 0 for r in reaction.products]):
                    total_propensity += self.get_propensity(reaction, reverse=True)
            time_end = time.time()
            print("Time tot-prop: ", time_end - time_start)
            print("tot-pro = ", total_propensity)
            ## drawing random numbers on uniform (0,1) distrubution
            r1 = random.random()
            r2 = random.random()
            ## Obtaining a time step tau from the probability distrubution
            ## P(t) = a*exp(-at) --> probability that any reaction occurs at time t
            tau = -np.log(r1) / total_propensity

            ## Choosing a reaction mu; need a cumulative sum of rxn propensities
            ## Discrete probability distrubution of reaction choice
            random_propensity = r2 * total_propensity
            time_start = time.time()
            prop_sum = 0
            for i, reaction in self.reactions.items():
                if all([self._state.get(r.entry_id, 0) > 0 for r in reaction.reactants]):
                    prop_sum += self.get_propensity(reaction, reverse=False)
                    if prop_sum > random_propensity:
                        reaction_mu = reaction
                        reverse = False
                        break
                if all([self._state.get(r.entry_id, 0) > 0 for r in reaction.products]):
                    prop_sum += self.get_propensity(reaction, reverse=True)
                    if prop_sum > random_propensity:
                        reaction_mu = reaction
                        reverse = True
                        break

            self.update_state(reaction_mu, reverse)
            time_end = time.time()
            print("Time to choose rxn and update state is: ", time_end - time_start)
            self.data["times"].append(tau)
            #self.data["reactions"].append({"reaction": reaction_mu, "reverse": reverse})

            t += tau
            #print(t)
            if reverse:
                for reactant in reaction_mu.products:
                    self.data["state"][reactant.entry_id].append((t,
                                                             self._state[reactant.entry_id]))
                for product in reaction_mu.reactants:
                    if product.entry_id not in self.data["state"]:
                        self.data["state"][product.entry_id] = [(0.0, 0),
                                                           (t, self._state[product.entry_id])]
                    else:
                        self.data["state"][product.entry_id].append((t,
                                                                self._state[product.entry_id]))

            else:
                for reactant in reaction_mu.reactants:
                    self.data["state"][reactant.entry_id].append((t,
                                                                  self._state[reactant.entry_id]))
                for product in reaction_mu.products:
                    if product.entry_id not in self.data["state"]:
                        self.data["state"][product.entry_id] = [(0.0, 0),
                                                                (t, self._state[product.entry_id])]
                    else:
                        self.data["state"][product.entry_id].append((t,
                                                                     self._state[product.entry_id]))


        for mol_id in self.data["state"]:
            self.data["state"][mol_id].append((t, self._state[mol_id]))

        return self.data

    def reaction_choice(self):
        """For the purposes of testing simulate() function, specifically the reaction choice functionality.
        """
        total_propensity = 0
        for i, reaction in self.reactions.items():
            if all([self._state.get(r.entry_id, 0) > 0 for r in reaction.reactants]):
                total_propensity += self.get_propensity(reaction, reverse=False)
            if all([self._state.get(r.entry_id, 0) > 0 for r in reaction.products]):
                total_propensity += self.get_propensity(reaction, reverse=True)
        random_propensity = random.random() * total_propensity

        prop_sum = 0
        for i, reaction in self.reactions.items():
            if all([self._state.get(r.entry_id, 0) > 0 for r in reaction.reactants]):
                prop_sum += self.get_propensity(reaction, reverse=False)
                if prop_sum > random_propensity:
                    reaction_mu = reaction
                    reverse = False
                    break
            if all([self._state.get(r.entry_id, 0) > 0 for r in reaction.products]):
                prop_sum += self.get_propensity(reaction, reverse=True)
                if prop_sum > random_propensity:
                    reaction_mu = reaction
                    reverse = True
                    break
        return (reaction_mu, reverse)

    def plot_trajectory(self, data=None, name=None, filename=None, num_label=10):
        """
        Plot KMC simulation data

        Args:
            data (dict): Dictionary containing output from a KMC simulation run.
                Default is None, meaning that the data stored in this
                ReactionPropagator object will be used.
            name(str): Title for the plot. Default is None.
            filename (str): Path for file to be saved. Default is None, meaning
                pyplot.show() will be used.
            num_label (int): Number of most prominent molecules to be labeled.
                Default is 10

        Returns:
            None
        """

        fig, ax = plt.subplots()

        if data is None:
            data = self.data

        # To avoid indexing errors
        if num_label > len(data["state"].keys()):
            num_label = len(data["state"].keys())

        # Sort by final concentration
        # We assume that we're interested in the most prominent products
        ids_sorted = sorted([(k, v) for k, v in data["state"].items()],
                            key=lambda x: x[1][-1][-1])
        ids_sorted = [i[0] for i in ids_sorted][::-1]
        print("top 10 species ids: ", ids_sorted)
        # Only label most prominent products
        for mol_id in data["state"]:
            ts = np.array([e[0] for e in data["state"][mol_id]])
            nums = np.array([e[1] for e in data["state"][mol_id]])
            if mol_id in ids_sorted[0:num_label]:
                for entry in self.reaction_network.entries_list:
                    if mol_id == entry.entry_id:
                        this_composition = self.reaction_network.entries_list[
                            mol_id].molecule.composition.alphabetical_formula
                        this_charge = self.reaction_network.entries_list[mol_id].molecule.charge
                        this_label = this_composition + " " + str(this_charge)
                        break

                ax.plot(ts, nums, label = this_label)
            else:
                ax.plot(ts, nums)
        if name is None:
            title = "KMC simulation, total time {}".format(data["times"][-1])
        else:
            title = name

        ax.set(title=title,
               xlabel="Time (s)",
               ylabel="# Molecules")

        ax.legend(loc='upper right', bbox_to_anchor=(1, 1),
                    ncol=3, fontsize="small")
        # ax.legend(loc='best', bbox_to_anchor=(0.45, -0.175),
        #           ncol=5, fontsize="small")


        # if filename is None:
        #     plt.show()
        # else:
        #     fig.savefig(filename, dpi=600)
        plt.savefig(filename)