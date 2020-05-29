# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, k, R, N_A, pi
import time


__author__ = "Ronald Kam, Evan Spotte-Smith"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"
__credit__ = "Xiaowei Xie"



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
        self.num_rxns = len(self.reaction_network.reactions)
        self.initial_state_conc = initial_state
        self.volume = volume
        self._state = dict()
        self.initial_state = dict()
        ## State will have number of molecules, instead of concentration
        for molecule_id, concentration in self.initial_state_conc.items():
            num_mols = int(concentration * self.volume * N_A  *1000) # volume in m^3
            self.initial_state[molecule_id] = num_mols
            self._state[molecule_id] = num_mols
        """Initial loop through all reactions in network: make arrays for initial propensity calculation. 
        The rate constant array [k1f k1r k2f k2r ... ], other arrays indexed in same fashion.
        Also create a "mapping" of each species to its respective reaction it is involved in, for future convenience"""
        self.reactions = dict()
        self.rate_constants = np.zeros(2*self.num_rxns)
        self.coord_array = np.zeros(2*self.num_rxns)
        self.rxn_ind = np.arange(2 * self.num_rxns)
        self.species_rxn_mapping = dict() ## associating reaction index to each molecule
        for id, reaction in enumerate(self.reaction_network.reactions):
            self.reactions[id] = reaction
            self.rate_constants[2*id] = reaction.rate_constant()["k_A"]
            self.rate_constants[2* id+1] = reaction.rate_constant()["k_B"]
            num_reactants_for = list()
            num_reactants_rev = list()
            for reactant in reaction.reactants:
                num_reactants_for.append(self.initial_state.get(reactant.entry_id, 0))
                if reactant.entry_id not in self.species_rxn_mapping:
                    self.species_rxn_mapping[reactant.entry_id] = [2*id]
                else:
                    self.species_rxn_mapping[reactant.entry_id].append(2*id)
            for product in reaction.products:
                num_reactants_rev.append(self.initial_state.get(product.entry_id, 0))
                if product.entry_id not in self.species_rxn_mapping:
                    self.species_rxn_mapping[product.entry_id] = [2 * id + 1]
                else:
                    self.species_rxn_mapping[product.entry_id].append(2 * id + 1)

            ## Obtain coordination value for forward reaction
            if len(reaction.reactants) == 1:
                self.coord_array[2*id] = num_reactants_for[0]
            elif (len(reaction.reactants) == 2) and (reaction.reactants[0] == reaction.reactants[1]):
                self.coord_array[2*id] = num_reactants_for[0] * (num_reactants_for[0] - 1)
            elif (len(reaction.reactants) == 2) and (reaction.reactants[0] != reaction.reactants[1]):
                self.coord_array[2 * id] = num_reactants_for[0] * num_reactants_for[1]
            else:
                raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
            # For reverse reaction
            if len(reaction.products) == 1:
                self.coord_array[2*id+1] = num_reactants_rev[0]
            elif (len(reaction.products) == 2) and (reaction.products[0] == reaction.products[1]):
                self.coord_array[2*id+1] = num_reactants_rev[0] * (num_reactants_rev[0] - 1)
            elif (len(reaction.products) == 2) and (reaction.products[0] != reaction.products[1]):
                self.coord_array[2 * id+1] = num_reactants_rev[0] * num_reactants_rev[1]
            else:
                raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
        self.propensity_array = np.multiply(self.rate_constants, self.coord_array)
        self.total_propensity = np.sum(self.propensity_array)
        print("Initial total propensity = ", self.total_propensity)
        self.data = {"times": list(),
                     "reactions": list(),
                     "state": dict()}

    @property
    def state(self):
        return self._state

    def get_coordination(self, reaction, reverse):
        """
        Calculate the coordination number for a particular reaction, based on the reaction type
        number of molecules for the reactants.

        Args:
            reaction (Reaction)
            reverse (bool): If True, give the propensity for the reverse
                reaction. If False, give the propensity for the forwards
                reaction.

        Returns:
            propensity (float)
        """

        #rate_constant = reaction.rate_constant()
        if reverse:
            #k = rate_constant["k_B"]
            num_reactants = len(reaction.products)
            reactants = reaction.products
        else:
            #k = rate_constant["k_A"]
            num_reactants = len(reaction.reactants)
            reactants = reaction.reactants

        num_mols_list = list()
        #entry_ids = list() # for testing
        for reactant in reactants:
            reactant_num_mols = self.state.get(reactant.entry_id, 0)
            num_mols_list.append(reactant_num_mols)
            #entry_ids.append(reactant.entry_id)
        if num_reactants == 1:
            h_prop = num_mols_list[0]
        elif (num_reactants == 2) and (reactants[0].entry_id == reactants[1].entry_id):
            h_prop = num_mols_list[0] * (num_mols_list[0] - 1) / 2
        elif (num_reactants == 2) and (reactants[0].entry_id != reactants[1].entry_id):
            h_prop = num_mols_list[0] * num_mols_list[1]
        else:
            raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
        #propensity = h_prop * self.rate_constants[reaction_ind]
        #propensity = h_prop * k
        return h_prop
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

    def alter_rxn_by_product(self, product_id, k_factor_change, reaction_classes = None):
        """Alter the rate constant of a reaction, based on the product(s) formed. For example, decreasing the rate
        constant for reactions that form undesired/unstable products. The change of k is directly proportional
        to the probability of reaction firing.

        Args:
            product (molecule entry id):
            k_magnitude_change (float): factor of change desired to be made to rate constant
            reaction_classes (list): type of reactions to consider

        Returns:
            altered self.rate_constants
        """
        # search for rxns that form product
        # change the corresponding rate constants based on the factor of change
        new_k = k * 298.15 / h * k_factor_change
        rxn_update = self.species_rxn_mapping[product_id]
        # Create the list of rxn ind to update k for
        for ind in rxn_update:
            this_rxn = self.reactions[math.floor(ind / 2)]
            if reaction_classes is None:
                if ind % 2:
                    self.rate_constants[ind - 1] = new_k
                else:
                    self.rate_constants[ind + 1] = new_k
            else:
                for rxn_class in reaction_classes:
                    if this_rxn.rate_type["class"] == rxn_class:
                        if ind % 2:
                            self.rate_constants[ind - 1] = new_k
                        else:
                            self.rate_constants[ind + 1] = new_k
        return self.rate_constants

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
                     "reaction_ids": list(),
                     "state": dict()}

        for mol_id in self._state.keys():
            self.data["state"][mol_id] = [(0.0, self._state[mol_id])]

        step_counter = 0
        while t < t_end:
            step_counter += 1

            ## Obtain reaction propensities, on which the probability distributions of
            ## time and reaction choice depends.

            #propensity_list = list()
            # for i, reaction in self.reactions.items():
            #     if all([self._state.get(r.entry_id, 0) > 0 for r in reaction.reactants]):
            #         rxn_ind = 2*i
            #         for_propensity = self.get_propensity(reaction, rxn_ind, reverse=False)
            #         propensity_array = np.append(propensity_array, for_propensity)
            #         rxn_ind_array = np.append(rxn_ind_array, rxn_ind)
            #         #propensity_list.append(for_propensity)
            #         total_propensity += for_propensity
            #     # else:
            #     #     propensity_array = np.append(propensity_array, 0)
            #     if all([self._state.get(r.entry_id, 0) > 0 for r in reaction.products]):
            #         rxn_ind = 2*i + 1
            #         rev_propensity = self.get_propensity(reaction, rxn_ind, reverse=True)
            #         propensity_array = np.append(propensity_array, rev_propensity)
            #         rxn_ind_array = np.append(rxn_ind_array, rxn_ind)
            #         #propensity_list.append(rev_propensity)
            #         total_propensity += rev_propensity
            #     # else:
            #     #     propensity_array = np.append(propensity_array, 0)

            ## drawing random numbers on uniform (0,1) distrubution
            r1 = random.random()
            r2 = random.random()
            ## Obtaining a time step tau from the probability distrubution
            ## P(t) = a*exp(-at) --> probability that any reaction occurs at time t
            tau = -np.log(r1) / self.total_propensity

            ## Choosing a reaction mu; need a cumulative sum of rxn propensities
            ## Discrete probability distrubution of reaction choice
            #time_start = time.time()
            random_propensity = r2 * self.total_propensity

            # prop_sum = 0
            # for i, reaction in self.reactions.items():
            #     if all([self._state.get(r.entry_id, 0) > 0 for r in reaction.reactants]):
            #         rxn_ind = 2 * i
            #         prop_sum += self.get_propensity(reaction, rxn_ind, reverse=False)
            #         if prop_sum > random_propensity:
            #             reaction_mu = reaction
            #             reverse = False
            #             break
            #     if all([self._state.get(r.entry_id, 0) > 0 for r in reaction.products]):
            #         rxn_ind = 2*i + 1
            #         prop_sum += self.get_propensity(reaction, rxn_ind, reverse=True)
            #         if prop_sum > random_propensity:
            #             reaction_mu = reaction
            #             reverse = True
            #             break
            reaction_choice_ind = self.rxn_ind[np.where(np.cumsum(self.propensity_array) >=  random_propensity)[0][0]]
            reaction_mu = self.reactions[math.floor(reaction_choice_ind / 2 )]

            if reaction_choice_ind % 2:
                reverse = True
            else:
                reverse = False
            #time_end = time.time()
            #print("Time to choose reaction = ", time_end - time_start)

            self.update_state(reaction_mu, reverse)
            #time_start = time.time()
            reactions_to_change = list()
            for reactant in reaction_mu.reactants:
                reactions_to_change.extend(self.species_rxn_mapping[reactant.entry_id])
            for product in reaction_mu.products:
                reactions_to_change.extend(self.species_rxn_mapping[product.entry_id])
            reactions_to_change = set(reactions_to_change)

            for rxn_ind in reactions_to_change:
                if rxn_ind % 2:
                    this_reverse = True
                else:
                    this_reverse = False
                this_h = self.get_coordination(self.reactions[math.floor(rxn_ind / 2)], this_reverse)
                self.coord_array[rxn_ind] = this_h

            self.propensity_array = np.multiply(self.rate_constants, self.coord_array)
            self.total_propensity = np.sum(self.propensity_array)
            #time_end = time.time()
            # print("Total prop = ", self.total_propensity)
            #print("Time to calculate total propensity = ", time_end - time_start)

            self.data["times"].append(tau)
            # self.data["reaction_ids"].append({"reaction": reaction_mu, "reverse": reverse})
            self.data["reaction_ids"].append(reaction_choice_ind)

            t += tau
            # print(t)
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
        print("top 15 species ids: ", ids_sorted[0:15])
        # Only label most prominent products
        colors = plt.cm.get_cmap('hsv', num_label)
        id = 0
        for mol_id in data["state"]:
            ts = np.array([e[0] for e in data["state"][mol_id]])
            nums = np.array([e[1] for e in data["state"][mol_id]])
            if mol_id in ids_sorted[0:num_label]:
                for entry in self.reaction_network.entries_list:
                    if mol_id == entry.entry_id:
                        this_composition = entry.molecule.composition.alphabetical_formula
                        this_charge = entry.molecule.charge
                        this_label = this_composition + " " + str(this_charge)
                        this_color = colors(id)
                        id +=1
                        #this_label = entry.entry_id
                        break

                ax.plot(ts, nums, label = this_label, color = this_color)
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
                    ncol=2, fontsize="small")
        # ax.legend(loc='best', bbox_to_anchor=(0.45, -0.175),
        #           ncol=5, fontsize="small")


        # if filename is None:
        #     plt.show()
        # else:
        #     fig.savefig(filename, dpi=600)
        if filename == None:
            plt.savefig("SimulationRun")
        else:
            plt.savefig(filename)

    def reaction_analysis(self, data = None):
        if data == None:
            data = self.data
        reaction_analysis_results = dict()
        reaction_analysis_results["endo_rxns"] = dict()
        rxn_count = np.zeros(2*self.num_rxns)
        endothermic_rxns_count = 0
        fired_reaction_ids = set(self.data["reaction_ids"])
        for ind in fired_reaction_ids:
            this_count = data["reaction_ids"].count(ind)
            rxn_count[ind] = this_count
            this_rxn = self.reactions[math.floor(ind/2)]
            if ind % 2: # reverse rxn
                if this_rxn.free_energy()["free_energy_B"] > 0: # endothermic reaction
                    endothermic_rxns_count += this_count
            else:
                if this_rxn.free_energy()["free_energy_A"] > 0: # endothermic reaction
                    endothermic_rxns_count += this_count
        reaction_analysis_results["endo_rxns"]["endo_count"] = endothermic_rxns_count
        sorted_rxn_ids = sorted(self.rxn_ind, key = lambda k: rxn_count[k], reverse = True)
        bar_rxns_labels = list()
        bar_rxns_count = list()
        bar_rxns_x = list()
        for i in range(15): # analysis on most frequent reactions
            this_rxn_id = sorted_rxn_ids[i]
            bar_rxns_x.append(str(this_rxn_id))
            this_reaction = self.reactions[math.floor(this_rxn_id / 2 )]
            reaction_analysis_results[this_rxn_id] = dict()
            reaction_analysis_results[this_rxn_id]["count"] = rxn_count[this_rxn_id]
            reaction_analysis_results[this_rxn_id]["reactants"] = list()
            reaction_analysis_results[this_rxn_id]["products"] = list()
            this_label = str()
            if this_rxn_id % 2: # reverse rxn
                reaction_analysis_results[this_rxn_id]["reaction_type"] = this_reaction.reaction_type()["rxn_type_B"]
                this_label += this_reaction.reaction_type()["rxn_type_B"]
                for reactant in this_reaction.products:
                    reaction_analysis_results[this_rxn_id]["reactants"].append((reactant.molecule.composition.alphabetical_formula , reactant.entry_id))
                    this_label += " " + reactant.molecule.composition.alphabetical_formula
                for product in this_reaction.reactants:
                    reaction_analysis_results[this_rxn_id]["products"].append((product.molecule.composition.alphabetical_formula , product.entry_id))
                    this_label += " " + product.molecule.composition.alphabetical_formula
                reaction_analysis_results[this_rxn_id]["rate constant"] = this_reaction.rate_constant()["k_B"]
            else: # forward rxn
                reaction_analysis_results[this_rxn_id]["reaction_type"] = this_reaction.reaction_type()["rxn_type_A"]
                this_label += this_reaction.reaction_type()["rxn_type_A"]
                for reactant in this_reaction.reactants:
                    reaction_analysis_results[this_rxn_id]["reactants"].append((reactant.molecule.composition.alphabetical_formula, reactant.entry_id))
                    this_label += " " + reactant.molecule.composition.alphabetical_formula
                for product in this_reaction.products:
                    reaction_analysis_results[this_rxn_id]["products"].append((product.molecule.composition.alphabetical_formula , product.entry_id))
                    this_label += " " + product.molecule.composition.alphabetical_formula
                reaction_analysis_results[this_rxn_id]["rate constant"] = this_reaction.rate_constant()["k_A"]
            bar_rxns_labels.append(this_label)
            bar_rxns_count.append(reaction_analysis_results[this_rxn_id]["count"])
        plt.figure()
        plt.bar(bar_rxns_x[:10], bar_rxns_count[:10])
        plt.xlabel("Reaction Index")
        plt.ylabel("Reaction Occurrence")
        plt.title("Top Reactions, total " + str(len(self.data["times"])) + " reactions")
        plt.savefig("li_limited_top_rxns")
        return reaction_analysis_results




