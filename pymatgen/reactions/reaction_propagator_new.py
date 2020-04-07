# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import logging
import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.analysis.reaction_network.reaction_network import ReactionNetwork
from pymatgen.reactions.reaction_rates import ReactionRateCalculator
from monty.serialization import dumpfn, loadfn
import random
import os
import matplotlib.pyplot as plt
from ase.units import eV, J, mol
import copy
import pickle


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

    def __init__(self, reaction_network):
        self.reaction_network = reaction_network
        self.reaction_list = reaction_network.reactions ### what to use as inputs for this function?
        ### should be list of reaction objects

        ### make a dict, assigning an index for each reaction. Each index is a dict containing reaction object and will later add propensity
        self.reactions = {}
        for id, reaction in enumerate(self.reaction_list):
            self.reactions[id] = reaction
        self.entries_dict = self.reaction_network.entries ### Dict containing all entries

    def reaction_propensities(self, state):
        """ Obtain reaction propensities, based on reaction kinetics. Propensity = (rate)(# distinct reactant combinations).
            Loop through all reactions, obtain rate constant

            Inputs:
            state (dict?) molecule entries, along with concentration of each molecule?

            Returns:
            (Num_rxns x 1) Dictionary of reaction propensities
        """
        propensities = {}
        for id, reaction in enumerate(self.reactions):
            rate = self.calculate_rate(self, state, temperature=298.0, reverse=False, kappa=1.0)
            reactants = reaction.reactants
            num_reactants = len(reactants)
            concentrations = []
            ## Find matching id, and get the concentration
            for this_reactant in reactants:
                reactant_id = this_reactant.entry_id
                for x in state:
                    if reactant_id == state[x]["Molecule"].entry_id: ## depends on data structure of "state"
                        concentrations = np.append(concentrations, state[x]["Concentration"])
                        break
            if num_reactants == 1:
                h = concentrations[0]
            elif (num_reactants == 2) & (reactants[0] == reactants[1]):
                h = concentrations[0] * (concentrations[0] - 1) / 2
            elif (num_reactants == 2) & (reactants[0] != reactants[1]):
                h = concentrations[0] * concentrations[1]
            else:
                raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
            propensities[id] = h * rate
        return propensities

    def update_state(self, state, reaction):
        """ Update the system based on the reaction chosen
        Parameters:
        state (dict) number of molecules of each species
        reaction object
        """
        num_species = len(state)
        reactant_ids = [r_id for r_id in reaction.reactant_ids]
        num_reactants = len(reactant_ids)
        product_ids = [p_id for p_id in reaction.product_ids]
        num_products = len(product_ids)
        reactants = reaction.reactants
        products = reaction.products

        unchecked_r_ind = [i for i in range(len(reactant_ids))]
        unchecked_p_ind = [j for j in range(len(product_ids))]
## Loop through state once, to find reactant and product ids
        for x in state:
            this_entryid = state[x]["Molecule"].entry_id
            if this_entryid in reactant_ids:
                state[x]["Concentration"] = state[x]["Concentration"] - 1
                unchecked_r_ind.remove[reactant_ids.index(this_entryid)]
            elif this_entryid in product_ids:
                state[x]["Concentration"] = state[x]["Concentration"] + 1
                unchecked_p_ind.remove[product_ids.index(this_entryid)]
            if (len(unchecked_p_ind) == 0) & len(unchecked_r_ind == 0):
                continue
                
        if len(unchecked_p_ind) > 0:
            for i, index in enumerate(unchecked_p_ind):
                state[num_species + i]["Molecule"] = products[index]
                state[num_species + i]["Concentration"] = 1
            num_species += len(unchecked_p_ind)

""" Alternative: loop through each reactant, then loop through state to find the id: more straightforward but needs more computation
        for reactant in reaction.reactants:
            r_id = reactant.entry_id
            for x in state:
                if r_id == state[x]["Molecule"].entry_id:
                    state[x]["Concentration"] = state[x]["Concentration"] - 1
                    break
        for product in reaction.products:
            p_id = product.entry_id
            existing_prod = False
            for y in state:
                if p_id == state[y]["Molecule"].entry_id:
                    state[y]["Concentration"] = state[y]["Concentration"] + 1
                    existing_prod = True
                    break
            if not existing_prod:
                num_species += 1
                state[num_species]["Molecule"] = product
                state[num_species]["Concentration"] = 1

"""

        return state

    def simulation(self, initial_state, t_end):
        """ Main body code of the KMC simulation. Propagates time and updates species amounts.
            Store reactions, time, and time step for each iteration

            Parameters
            state: (dict) number of molecules of each species; need to know indexing of molecules
            t_end: (int) ending time of simulation

            Returns
            final state of molecules
        """
        state = initial_state
        t = 0
        data = {}
        data["Time"] = []
        data["Tau"] = []
        data["Reactions"] = []
        while t < t_end:
            ## Obtain reaction propensities, on which the probability distributions of
            ## time and reaction choice depends.
            propensities = self.reaction_propensities(self, state) ## dict of each reaction's propensity
            a = 0
            ## Loop to sum up all propensities
            for i in propensities:
                a = a + propensities[i]

            ## drawing random numbers on uniform (0,1) distrubution
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()

            ## Obtaining a time step tau from the probability distrubution
            ## P(t) = a*exp(-at) --> probability that any reaction occurs at time t
            tau = -np.log(r1)/a
            t = t + tau
            ## Choosing a reaction mu; need a cumulative sum of rxn propensities
            ## Discrete probability distrubution of reaction choice
            prop_sum = 0
            for i in propensities:
                prop_sum = prop_sum + propensities[i]
                if prop_sum > r2*a:
                    mu = i
                    break
            reaction_mu = self.reactions[mu]
            state = update_state(self, state, reaction_mu) ## updated molecule amounts
            data["Time"] = np.append(data["Time"], t)
            data["Tau"] = np.append(data["Tau"], tau)
            data["Reactions"] = np.append(data["Reactions"], reaction_mu)
