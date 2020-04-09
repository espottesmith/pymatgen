# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import logging
import numpy as np

__author__ = "Ronald Kam"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "1.0"

k_b = 1.38064852e-23
T = 298.15
h = 6.62607015e-34
R = 8.3144598

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
        2) initial_state: (dict) {Molecule ID: {"molecule": MoleculeEntry, "num_mols": (int)}}
        """
        self.reaction_network = reaction_network
        ### make a dict, assigning an index for each reaction. Each index is a dict containing reaction object and will later add propensity
        self.reactions = dict()
        for id, reaction in enumerate(self.reaction_network.reactions):
            self.reactions[id] = reaction
        self.initial_state = initial_state
        self.state = initial_state
    def reaction_propensities(self):
        """ Obtain reaction propensities, based on reaction kinetics. Propensity = (rate)(# distinct reactant combinations).
            Loop through all reactions, obtain rate constant
            Returns:
            (Num_rxns x 1) Dictionary of reaction propensities
        """
        propensities = dict()
        for id, reaction in enumerate(self.reactions):
            k = reaction.calculate_rate_constant(self, state, temperature=298.0, reverse=False, kappa=1.0)
            reactants = reaction.reactants
            num_reactants = len(reactants)
            concentrations = list()
            ## Find matching id, and get the concentration
            for this_reactant in reactants:
                reactant_id = this_reactant.entry_id
                mol_dict = state.get(reactant_id, False)
                if mol_dict is 0:
                    raise RunTimeError("This reaction cannot occur, as the reactants aren't present")
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
        data["Time"] = list()
        data["Tau"] = list()
        data["Reactions"] = list()
        while t < t_end:
            ## Obtain reaction propensities, on which the probability distributions of
            ## time and reaction choice depends.
            propensities = self.reaction_propensities() ## dict of each reaction's propensity
            a = 0
            ## Loop to sum up all propensities
            for i in propensities:
                a += propensities[i]

            ## drawing random numbers on uniform (0,1) distrubution
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()

            ## Obtaining a time step tau from the probability distrubution
            ## P(t) = a*exp(-at) --> probability that any reaction occurs at time t
            tau = -np.log(r1)/a
            t += tau
            ## Choosing a reaction mu; need a cumulative sum of rxn propensities
            ## Discrete probability distrubution of reaction choice

            ## A check to see if reactants are actually present in current state. If yes, then will proceed with that reaction
            reactants_present = False
            while reactants_present is False
                prop_sum = 0
                for i in propensities:
                    prop_sum += propensities[i]
                    if prop_sum > r2*a:
                        mu = i
                        break
                reaction_chosen = self.reactions[mu]
                reactants_present = all([reactant_id in state for reactant_id in reaction_chosen.reactant_ids])
            reaction_mu = reaction_chosen
            self.state = update_state(self, reaction_mu) ## updated molecule amounts
            data["Time"].append(t)
            data["Tau"].append(tau)
            data["Reactions"].append(reaction_mu)

        return data
