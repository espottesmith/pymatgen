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


    def reaction_propensities(self, state):
        """ Obtain reaction propensities, based on reaction kinetics. Propensity = (rate)(orientation factor).
            Loop through all reaction nodes, obtain rate constant

            Inputs:
            self - need all reaction nodes
            state (array) number of molecules of each species

            Returns:
            (Num_rxns x 1) Array of reaction propensities
        """

        return propensities


    def simulation(self, initial_state, t_end):
        """ Main body code of the KMC simulation. Propagates time and updates species amounts.
            Need to store reactions, concentrations at each time step

            Parameters
            state: (array) number of molecules of each species; need to know indexing of molecules
            t_end: (int) ending time of simulation

            Returns
            final state of molecules
        """
        state = initial_state
        t = 0
        while t < t_end:
            current_state = state[-1,:]
            ## Obtain reaction propensities, on which the probability distributions of
            ## time and reaction choice depends.
            propensities = self.reaction_propensities(self, current_state) ## should be an array of each reaction's propensity
            a = np.sum(propensities)
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
            for i, ai in enumerate(propensities):
                prop_sum = prop_sum + ai
                if prop_sum > r2*a:
                    mu = i
                    break

            ## Next work on updating state based on choice of reaction
            ## relate rxn index mu to the actual reaction node, and update state

            new_state = [] ## updated molecule amounts
            state = np.vstack((state, new_state)) ## Store concentrations at each time step
