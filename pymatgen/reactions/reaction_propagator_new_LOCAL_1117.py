# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import numpy as np


__author__ = "Ronald Kam"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "1.0"

k_b = 1.38064852e-23
T = 298.15
h = 6.62607015e-34
R = 8.3144598
N = 6.0221409e+23

class ReactionPropagator:
    """
    Class for stochastic simulation of a given reaction network. Method is described by Gillespie (1976)
    Main functions:
    - Time step and reaction choice
        - Requirements:
            - List of reactions
            - Reaction propensities from rates
    - Update state of molecules
    """

    def __init__(self, reaction_network, initial_state):
        """ Args:
        1) Reaction Network object
        2) initial_state: (dict) {Molecule ID: {"molecule": MoleculeEntry, "concentration": (int)}}
        """
        self.reaction_network = reaction_network
        ### make a dict, assigning an index for each reaction. Each index is a dict containing reaction object and will later add propensity
        self.reactions = dict()
        for id, reaction in enumerate(self.reaction_network.reactions):
            self.reactions[id] = reaction
        self.initial_state = initial_state
        self.volume = 1  #L
        self.state = dict()
        ## State will have number of molecules, instead of concentration
        for molecule_id in initial_state:
            num_mols = molecule_id["concentration"] * self.volume * N
            self.state[molecule_id] = dict()
            self.state[molecule_id]["molecule"] = initial_state[molecule_id]["molecule"]
            self.state[molecule_id]["num_mols"] = initial_state[molecule_id]["concentration"] * V * N

    @property
    def state(self):
        return self.state

    def get_propensity(self, reaction, reverse):
        "Obtain reaction propensity for a single reaction"
        k = reaction.calculate_rate_constant(self, temperature = 298.0, reverse, kappa = 1.0)
        num_reactants = len(reaction.reactants)
        num_mols_list = list()
        for reactant in reaction.reactants:
            reactant_num_mols = self.state.get(reactant.entry_id).get("num_mols")
            num_mols_list.append(reactant_num_mols)
        if num_reactants == 1:
            h = num_mols_list[0]
        elif (num_reactants == 2) and (reactants[0] == reactants[1]):
            h = num_mols_list[0] * (num_mols_list[0] - 1) / 2
        elif (num_reactants == 2) and (reactants[0] != reactants[1]):
            h = num_mols_list[0] * num_mols_list[1]
        else:
            raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
        propensity = h * k
        return propensity

    def reaction_propensities(self):
        """ Obtain reaction propensities, based on reaction kinetics. Propensity = (rate)(# distinct reactant combinations).
            Loop through all reactions, obtain rate constant
            Returns:
            (Num_rxns x 1) Dictionary of reaction propensities
        """
        propensities = dict()
        for id, reaction in enumerate(self.reactions):
            k = reaction.calculate_rate_constant(self, temperature=298.0, reverse, kappa=1.0)
            num_reactants = len(reaction.reactants)
            concentrations = list()
            ## Find matching id, and get the concentration
            for this_reactant in reaction.reactants:
                reactant_id = this_reactant.entry_id
                mol_dict = state.get(reactant_id)
            if num_reactants == 1:
                h = concentrations[0]
            elif (num_reactants == 2) & (reactants[0] == reactants[1]):
                h = concentrations[0] * (concentrations[0] - 1) / 2
            elif (num_reactants == 2) & (reactants[0] != reactants[1]):
                h = concentrations[0] * concentrations[1]
            else:
                raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
            propensities[id] = h * k
        return propensities

    def update_state(self, reaction):
        """ Update the system based on the reaction chosen
        Parameters:
        state (dict) number of molecules of each species
        reaction object
        """
        for r_id in reaction.reactant_ids:
            self.state[r_id]["num_mols"] -=  1
        for product in reaction.products:
            p_id = product.entry_ids
            if p_id in self.state:
                self.state[p_id]["num_mols"] +=1
            else:
                self.state[p_id]["molecule"] = product
                self.state[p_id]["num_mols"] = 1
        return self.state

    def simulation(self, t_end):
        """ Main body code of the KMC simulation. Propagates time and updates species amounts.
            Store reactions, time, and time step for each iteration

            Parameters
            state: (dict) number of molecules of each species; need to know indexing of molecules
            t_end: (int) ending time of simulation

            Returns
            final state of molecules
        """
        self.state = self.initial_state
        t = 0
        data = dict()
        data["time"] = list()
        data["tau"] = list()
        data["reactions"] = dict()
        step_counter = 0
        while t < t_end:
            step_counter += 1
            ## Obtain reaction propensities, on which the probability distributions of
            ## time and reaction choice depends.
            total_propensity = 0
            for reaction in self.reaction_network.reactions:
                total_propensity += self.get_propensity(reaction, reverse = False) + self.get_propensity(reaction, reverse = True)

            ## drawing random numbers on uniform (0,1) distrubution
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()
            random_propensity = r2 * total_propensity
            ## Obtaining a time step tau from the probability distrubution
            ## P(t) = a*exp(-at) --> probability that any reaction occurs at time t
            tau = -np.log(r1)/a
            t += tau
            ## Choosing a reaction mu; need a cumulative sum of rxn propensities
            ## Discrete probability distrubution of reaction choice
            prop_sum = 0
            for i, reaction in enumerate(self.reaction_network.reactions):
                prop_sum += self.get_propensity(reaction, reverse = False)
                if prop_sum > random_propensity:
                    reaction_mu = self.reactions[i]
                    reverse = False
                    break
                prop_sum += self.get_propensity(reaction, reverse = True)
                if prop_sum > random_propensity:
                    reaction_mu = self.reaction_network.reactions[i]
                    reverse = True
                    break
            self.state = update_state(self, reaction_mu) ## updated molecule amounts
            data["time"].append(t)
            data["tau"].append(tau)
            data["reactions"][step_counter] = {"reaction_object" : reaction_mu, "reverse": reverse}

        return data
