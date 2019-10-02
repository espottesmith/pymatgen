# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import json

from monty.json import MontyEncoder, MontyDecoder

from monty.json import MSONable
from pymatgen.analysis.reaction_rates import (ReactionRateCalculator,
                                              BEPRateCalculator,
                                              ExpandedBEPRateCalculator)

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
                 alpha=None, kappa=1.0, parameters=None, entry_id=None,
                 attribute=None):
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
                reaction kinetics. This reference NEEDS TO HAVE A TRANSITION
                STATE OF ITS OWN.
            approximate_method (str): If transition_state is None (default),
                some method must be used to estimate the reaction kinetics.
                By default, this is "EBEP", meaning the
                ExpandedBEPRateCalculator will be used. "BEP", for
                BEPRateCalculator, is also a valid option.
            alpha (float): Reaction coordinate for the reaction (default is
                None, meaning that some assumptions will be made)
            kappa (float): Depending on the rate calculator used, kappa
                represents either the transmission coefficient or the steric
                factor. In both cases, the meaning is similar; a low kappa
                indicates that a reaction is not likely to happen, while a high
                kappa indicates that the reaction is likely to happen. Default
                is 1.0.
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

        self.parameters = parameters if parameters else dict()
        self.entry_id = entry_id
        self.attribute = attribute

        if alpha is None:
            self.alpha = 0.5
        else:
            self.alpha = alpha

        if kappa is None:
            self.kappa = 1.0
        else:
            self.kappa = kappa

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
                    ea = self.reference_reaction.rate.calculate_act_energy()
                    delta_h = self.reference_reaction.rate.net_enthalpy

                    self.rate = BEPRateCalculator(self.reactants,
                                                  self.products,
                                                  ea,
                                                  delta_h,
                                                  alpha=self.alpha)
                elif approximate_method.upper() == "EBEP":
                    ea = self.reference_reaction.rate.calculate_act_energy()
                    ha = self.reference_reaction.rate.calculate_act_enthalpy()
                    sa = self.reference_reaction.rate.calculate_act_entropy()
                    delta_e = self.reference_reaction.rate.net_energy
                    delta_h = self.reference_reaction.rate.net_enthalpy
                    delta_s = self.reference_reaction.rate.net_entropy

                    self.rate = ExpandedBEPRateCalculator(self.reactants,
                                                          self.products,
                                                          ea, ha, sa,
                                                          delta_e,
                                                          delta_h,
                                                          delta_s,
                                                          alpha=self.alpha)
                else:
                    # This should never be reached
                    self.rate = None
        else:
            self.rate = ReactionRateCalculator(self.reactants,
                                               self.products,
                                               self.transition_state)

        self.reaction = self.rate.reaction
        self.delta_e = self.rate.net_energy
        self.delta_h = self.rate.net_enthalpy
        self.delta_s = self.rate.net_entropy

    @property
    def reaction_string(self):
        """
        Return a string representation of the reaction

        Args:
            None

        Returns:
            str()
        """

        return str(self.reaction)

    def delta_g(self, temperature=298.0):
        """
        Calculate net reaction Gibbs free energy at a given temperature.

        ΔG = ΔH - T ΔS

        Args:
            temperature (float): absolute temperature in Kelvin

        Returns:
            float: net Gibbs free energy (in eV)
        """

        return self.rate.calculate_net_gibbs(temperature=temperature)

    def rate_constant(self, temperature=298.0):
        """
        Calculate the rate constant k by the Eyring-Polanyi equation of
        transition state theory (possible if the transition state is known, or
        if the EBEP method is being used), or by collision theory (if BEP is
        being used).

        Args:
            temperature (float): absolute temperature in Kelvin

        Returns:
            k_rate (float): temperature-dependent rate constant
        """

        return self.rate.calculate_rate_constant(temperature=temperature,
                                                 kappa=self.kappa)

    def reaction_rate(self, concentrations, temperature=298.0):
        """
        Calculate the based on the reaction stoichiometry.

        NOTE: Here, we assume that the reaction is an elementary step.

        Args:
            concentrations (list): concentrations of reactant molecules.
                Order of the reactants DOES matter.
            temperature (float): absolute temperature in Kelvin

        Returns:
            rate (float): reaction rate, based on the stoichiometric rate law
                and the rate constant
        """

        return self.rate.calculate_rate(concentrations, temperature=temperature,
                                        kappa=self.kappa)

    def __repr__(self):
        name = "ReactionEntry {}".format(self.entry_id)
        rxn_str = "Reaction: {}".format(self.reaction_string)
        rct_str = " + ".join([r.molecule.composition.alphabetical_formula for r in self.reactants])
        pro_str = " + ".join([p.molecule.composition.alphabetical_formula for p in self.products])
        from_alpha = "With Alphabetical Formulas: {} --> {}".format(rct_str, pro_str)
        thermo_head = "Net Thermodynamic Properties:"
        thermo_vals = {"Energy": self.delta_e,
                       "Enthalpy": self.delta_h,
                       "Entropy": self.delta_s,
                       "Gibbs Free Energy (T=298.0K)": self.delta_g()}
        thermo_body = "\n".join(["\t{}: {}".format(k, v) for k, v in thermo_vals.items()])
        kinetic_head = "Standard Kinetic Properties:"
        kinetic_body = "Rate Constant (T=298.0K): {}".format(self.rate_constant())

        return "\n".join([name, rxn_str, from_alpha, thermo_head, thermo_body,
                          kinetic_head, kinetic_body])

    def __str__(self):
        return self.__repr__()