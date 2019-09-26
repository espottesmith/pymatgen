# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import json

from monty.json import MontyEncoder, MontyDecoder

from pymatgen.core.composition import Composition
from monty.json import MSONable
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.reaction_rates import (ReactionRateCalculator,
                                              BEPRateCalculator,
                                              ExpandedBEPRateCalculator)
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.analysis.fragmenter import metal_edge_extender

"""

"""


__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "September 2019"


class ReactionEntry(MSONable):
    """

    """

    def __init__(self, reactants, products, transition_state=None,
                 reference_reaction=None, approximate_method="EBEP",
                 parameters=None, entry_id=None, attribute=None):
        """
        An object which represents a chemical reaction (in terms of reactants,
        transition state, and products)

        Args:
            reactants (list): list of MoleculeEntry objects
            products (list): list of MoleculeEntry objects
            transition_state (MoleculeEntry): transition state for the reaction.
                Default is None, meaning all calculations (for instance,
                for kinetics) will be approximated.
            reference_reaction (ReactionEntry): if transition_state is None
                (default), another reaction must be used to estimate the
                reaction kinetics. This reference need not have a transition
                state of its own, technically, but note that at some point a
                ReactionEntry must have a transition state for any
                approximations to be made.
            approximate_method (str): If transition_state is None (default),
                some method must be used to estimate the reaction kinetics.
                By default, this is "EBEP", meaning the
                ExpandedBEPRateCalculator will be used. "BEP", for
                BEPRateCalculator, is also a valid option.
            parameters (dict): An optional dict of parameters associated with
                the reaction. Defaults to None.
            entry_id (obj): An optional id to uniquely identify the entry.
            attribute: An optional attribute of the entry. This can be used to
                specify, for instance, a particular label, or an application, or
                else ... Used for further analysis and plotting purposes.
                An attribute can be anything that is MSONable.

        Returns:
            None
        """

        self.reactants = reactants
        self.products = products

        self.transition_state = transition_state
        self.reference_reaction = reference_reaction

        if approximate_method in ["BEP", "EBEP"]:
            self.approximate_method = approximate_method
        else:
            raise ValueError("Only acceptable values for approximate_method are"
                             " 'BEP' and 'EBEP'!")

        if transition_state is None:
            if reference_reaction is None:
                raise ValueError("Cannot have transition_state == None and "
                                 "reference_reaction == None!")
            else:
                if approximate_method.upper() == "BEP":
                    ea = self.reference_reaction.calculator.calculate_act_energy
                    delta_h = self.reference_reaction.calculator.net_enthalpy
                    try:
                        alpha = self.reference_reaction.calculator.alpha
                    except AttributeError:
                        alpha = 0.5

                    self.calculator = BEPRateCalculator(self.reactants,
                                                        self.products,
                                                        ea,
                                                        delta_h,
                                                        alpha=alpha)
                elif approximate_method.upper() == "EBEP":
                    ga = self.reference_reaction.calculator.calculate_act_gibbs

                else:
                    # This should never be reached
                    self.calculator = None
        else: