from abc import ABCMeta, abstractproperty, abstractmethod, abstractclassmethod
from abc import ABC, abstractmethod
from gunicorn.util import load_class

import logging
import copy
import itertools
import heapq

import numpy as np
from scipy.constants import h, k, R, N_A, pi
import networkx as nx
from networkx.readwrite import json_graph
import networkx.algorithms.isomorphism as iso
from monty.json import MSONable, MontyDecoder

from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
from networkx.algorithms import bipartite
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.core.composition import CompositionError
from pymatgen.reactions.reaction_rates import (ReactionRateCalculator,
                                               ExpandedBEPRateCalculator)


def categorize(reaction, classes, templates, environment, charge):
    """
    Given reactants, products, and a local bonding environment, place a
        reaction into a reaction class.

    Note: This is not designed for redox reactions

    Args:
        reaction: Reaction object
        classes: dict of dicts representing families of reactions
        environment: a nx.Graph object representing a submolecule that
            defines the type of reaction
        templates: list of nx.Graph objects that define other classes
        charge: int representing the charge of the reaction
    Returns:
        classes: nested dict containing categorized reactions
        templates: list of graph representations of molecule "templates"
    """

    nm = iso.categorical_node_match("specie", "ERROR")

    match = False
    bucket_templates = copy.deepcopy(templates)

    for e, template in enumerate(bucket_templates):
        if nx.is_isomorphic(environment, template, node_match=nm):
            match = True
            label = e
            if label in classes[charge]:
                classes[charge][label].append(reaction)
            else:
                classes[charge][label] = [reaction]
            break
    if not match:
        label = len(templates)
        classes[charge][label] = [reaction]

        templates.append(environment)

    return classes, templates


class Reaction(MSONable, metaclass=ABCMeta):
    """
       Abstract class defining reactions.

       Args:
           reactants ([MoleculeEntry]): A list of MoleculeEntry objects of len 1.
           products ([MoleculeEntry]): A list of MoleculeEntry objects of max len 2
           transition_state (MoleculeEntry): MoleculeEntry representing the
               transition state of the reaction.
       """

    def __init__(self, reactants, products, transition_state=None, parameters=None):
        self.reactants = reactants
        self.products = products
        self.transition_state = transition_state
        if self.transition_state is None:
            # Provide no reference initially
            self.rate_calculator = ExpandedBEPRateCalculator(reactants, products,
                                                             0.0, 0.0, 0.0,
                                                             0.0, 0.0, 0.0,
                                                             alpha=-1.0)
        else:
            self.rate_calculator = ReactionRateCalculator(reactants, products,
                                                          self.transition_state)
        self.reactant_ids = list({e.entry_id for e in self.reactants})
        self.product_ids = list({e.entry_id for e in self.products})
        self.parameters = parameters or dict()

    def update_calculator(self, transition_state=None, reference=None):
        """
        Update the rate calculator with either a transition state (or a new
            transition state) or the thermodynamic properties of a reaction

        Args:
            transition_state (MoleculeEntry): MoleculeEntry referring to a
                transition state
            reference (dict): Dictionary containing relevant thermodynamic
                values for a reference reaction
                Keys:
                    delta_ea: Activation energy
                    delta_ha: Activation enthalpy
                    delta_sa: Activation entropy
                    delta_e: Reaction energy change
                    delta_h: Reaction enthalpy change
                    delta_s: Reaction entropy change
        Returns:
            None
        """

        if transition_state is None:
            if reference is None:
                pass
            else:
                self.rate_calculator = ExpandedBEPRateCalculator(
                    reactants=self.reactants,
                    products=self.products,
                    delta_ea_reference=reference["delta_ea"],
                    delta_ha_reference=reference["delta_ha"],
                    delta_sa_reference=reference["delta_sa"],
                    delta_e_reference=reference["delta_e"],
                    delta_h_reference=reference["delta_h"],
                    delta_s_reference=reference["delta_s"],
                )
        else:
            self.rate_calculator = ReactionRateCalculator(self.reactants, self.products,
                                                          transition_state)

    def __in__(self, entry):
        return entry.entry_id in self.entry_ids

    def __len__(self):
        return len(self.reactants)

    @classmethod
    @abstractmethod
    def generate(cls, entries):
        pass

    @abstractmethod
    def reaction_type(self):
        pass

    @abstractmethod
    def energy(self):
        pass

    @abstractmethod
    def free_energy(self):
        pass

    @abstractmethod
    def rate_constant(self):
        pass

    def as_dict(self) -> dict:
        if self.transition_state is None:
            ts = None
        else:
            ts = self.transition_state.as_dict()

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "reactants": [r.as_dict() for r in self.reactants],
             "products": [p.as_dict() for p in self.products],
             "transition_state": ts,
             "rate_calculator": self.rate_calculator.as_dict(),
             "parameters": self.parameters}

        return d

    @classmethod
    def from_dict(cls, d):
        reactants = [MoleculeEntry.from_dict(r) for r in d["reactants"]]
        products = [MoleculeEntry.from_dict(p) for p in d["products"]]
        if d["transition_state"] is None:
            ts = None
            rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactants, products, transition_state=ts,
                       parameters=parameters)
        reaction.rate_calculator = rate_calculator
        return reaction


class RedoxReaction(Reaction):

    """
    A class to define redox reactions as follows:
    One electron oxidation / reduction without change to bonding
        A^n ±e- <-> A^n±1
        Two entries with:
        identical composition
        identical number of edges
        a charge difference of 1
        isomorphic molecule graphs

    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of single molecular entry
       transition_state (MoleculeEntry): MoleculeEntry representing the
           transition state of the reaction.
    """

    def __init__(self, reactant, product, transition_state=None, parameters=None):
        if len(reactant) != 1 or len(product) != 1:
            raise RuntimeError("One electron redox requires two lists that each contain one entry!")
        self.reactant = reactant[0]
        self.product = product[0]
        self.electron_free_energy = None
        super().__init__([self.reactant], [self.product],
                         transition_state=transition_state,
                         parameters=parameters)

    @classmethod
    def generate(cls, entries):
        reactions = list()
        classes = dict()
        for formula in entries:
            classes[formula] = dict()
            for Nbonds in entries[formula]:
                charges = sorted(list(entries[formula][Nbonds].keys()))
                for charge in charges:
                    classes[formula][charge] = list()
                if len(charges) > 1:
                    for ii in range(len(charges) - 1):
                        charge0 = charges[ii]
                        charge1 = charges[ii + 1]
                        if charge1 - charge0 == 1:
                            for entry0 in entries[formula][Nbonds][charge0]:
                                for entry1 in entries[formula][Nbonds][charge1]:
                                    if entry0.mol_graph.isomorphic_to(entry1.mol_graph):
                                        r = cls([entry0], [entry1])
                                        reactions.append(r)
                                        classes[formula][charge0].append(r)

        return reactions, classes

    def reaction_type(self):
        val0 = self.reactant.charge
        val1 = self.product.charge
        if val1 < val0:
            rxn_type_A = "One electron reduction"
            rxn_type_B = "One electron oxidation"
        else:
            rxn_type_A = "One electron oxidation"
            rxn_type_B = "One electron reduction"

        reaction_type = {"class": "RedoxReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self):
        entry0 = self.reactant
        entry1 = self.product
        if entry1.free_energy() is not None and entry0.free_energy() is not None:
            free_energy_A = entry1.free_energy() - entry0.free_energy()
            free_energy_B = entry0.free_energy() - entry1.free_energy()

            if self.reaction_type()["rxn_type_A"] == "One electron reduction":
                free_energy_A += -self.electron_free_energy
                free_energy_B += self.electron_free_energy
            else:
                free_energy_A += self.electron_free_energy
                free_energy_B += -self.electron_free_energy
        else:
            free_energy_A = None
            free_energy_B = None
        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self):
        entry0 = self.reactant
        entry1 = self.product
        if entry1.energy is not None and entry0.energy is not None:
            energy_A = entry1.energy - entry0.energy
            energy_B = entry0.energy - entry1.energy
        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate_constant(self):
        """
        For now, all redox reactions will have the same
        """
        return 10.0 ** 11

    def as_dict(self) -> dict:
        if self.transition_state is None:
            ts = None
        else:
            ts = self.transition_state.as_dict()

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "reactants": [r.as_dict() for r in self.reactants],
             "products": [p.as_dict() for p in self.products],
             "reactant": self.reactant.as_dict(),
             "product": self.product.as_dict(),
             "electron_free_energy": self.electron_free_energy,
             "transition_state": ts,
             "rate_calculator": self.rate_calculator.as_dict(),
             "parameters": self.parameters}

        return d

    @classmethod
    def from_dict(cls, d):
        reactants = [MoleculeEntry.from_dict(r) for r in d["reactants"]]
        products = [MoleculeEntry.from_dict(p) for p in d["products"]]
        if d["transition_state"] is None:
            ts = None
            rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactants, products, transition_state=ts,
                       parameters=parameters)
        reaction.rate_calculator = rate_calculator
        reaction.electron_free_energy = d["electron_free_energy"]
        return reaction


class IntramolSingleBondChangeReaction(Reaction):

    """
    A class to define intramolecular single bond change as follows:
        Intramolecular formation / breakage of one bond
        A^n <-> B^n
        Two entries with:
            identical composition
            number of edges differ by 1
            identical charge
            removing one of the edges in the graph with more edges yields a graph isomorphic to the other entry

    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of single molecular entry
    """

    def __init__(self, reactant, product, transition_state=None, parameters=None):
        if len(reactant) != 1 or len(product) != 1:
            raise RuntimeError("Intramolecular single bond change requires two lists that each contain one entry!")
        self.reactant = reactant[0]
        self.product = product[0]
        super().__init__([self.reactant], [self.product],
                         transition_state=transition_state,
                         parameters=parameters)

    @classmethod
    def generate(cls, entries):
        reactions = list()
        classes = dict()
        templates = list()
        for formula in entries:
            Nbonds_list = list(entries[formula].keys())
            if len(Nbonds_list) > 1:
                for ii in range(len(Nbonds_list) - 1):
                    Nbonds0 = Nbonds_list[ii]
                    Nbonds1 = Nbonds_list[ii + 1]
                    if Nbonds1 - Nbonds0 == 1:
                        for charge in entries[formula][Nbonds0]:
                            if charge not in classes:
                                classes[charge] = dict()
                            if charge in entries[formula][Nbonds1]:
                                for entry1 in entries[formula][Nbonds1][charge]:
                                    for edge in entry1.edges:
                                        mg = copy.deepcopy(entry1.mol_graph)
                                        mg.break_edge(edge[0], edge[1], allow_reverse=True)
                                        if nx.is_weakly_connected(mg.graph):
                                            for entry0 in entries[formula][Nbonds0][charge]:
                                                if entry0.mol_graph.isomorphic_to(mg):
                                                    r = cls([entry0], [entry1])
                                                    reactions.append(r)
                                                    indices = entry1.mol_graph.extract_bond_environment([edge])
                                                    subg = entry1.graph.subgraph(list(indices)).copy().to_undirected()

                                                    classes, templates = categorize(r, classes, templates, subg, charge)
                                                    break

        return reactions, classes

    def reaction_type(self):
        val0 = self.reactant.charge
        val1 = self.product.charge
        if val1 < val0:
            rxn_type_A = "Intramolecular single bond breakage"
            rxn_type_B = "Intramolecular single bond formation"
        else:
            rxn_type_A = "Intramolecular single bond formation"
            rxn_type_B = "Intramolecular single bond breakage"

        reaction_type = {"class": "IntramolSingleBondChangeReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self):
        entry0 = self.reactant
        entry1 = self.product
        if entry1.free_energy() is not None and entry0.free_energy() is not None:
            free_energy_A = entry1.free_energy() - entry0.free_energy()
            free_energy_B = entry0.free_energy() - entry1.free_energy()
        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self):
        entry0 = self.reactant
        entry1 = self.product
        if entry1.energy is not None and entry0.energy is not None:
            energy_A = entry1.energy - entry0.energy
            energy_B = entry0.energy - entry1.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate_constant(self):
        if isinstance(self.rate_calculator, ReactionRateCalculator):
            return {"k_A": self.rate_calculator.calculate_rate_constant(),
                    "k_B": self.rate_calculator.calculate_rate_constant(reverse=True)}
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            # No reference is set
            # Use barrierless reaction
            if self.rate_calculator.alpha == -1:
                rate_constant = dict()
                free_energy = self.free_energy()

                if free_energy["free_energy_A"] < 0:
                    rate_constant["k_A"] = k * 298.15 / h
                else:
                    rate_constant["k_A"] = k * 298.15 / h * np.exp(-1 * free_energy["free_energy_A"] * 96487 / (R * 298.15))

                if free_energy["free_energy_B"] < 0:
                    rate_constant["k_B"] = k * 298.15 / h
                else:
                    rate_constant["k_B"] = k * 298.15 / h * np.exp(-1 * free_energy["free_energy_B"] * 96487 / (R * 298.15))

                return rate_constant
            else:
                return {"k_A": self.rate_calculator.calculate_rate_constant(),
                        "k_B": self.rate_calculator.calculate_rate_constant(reverse=True)}

    def as_dict(self) -> dict:
        if self.transition_state is None:
            ts = None
        else:
            ts = self.transition_state.as_dict()

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "reactants": [r.as_dict() for r in self.reactants],
             "products": [p.as_dict() for p in self.products],
             "reactant": self.reactant.as_dict(),
             "product": self.product.as_dict(),
             "transition_state": ts,
             "rate_calculator": self.rate_calculator.as_dict(),
             "parameters": self.parameters}

        return d

    @classmethod
    def from_dict(cls, d):
        reactants = [MoleculeEntry.from_dict(r) for r in d["reactants"]]
        products = [MoleculeEntry.from_dict(p) for p in d["products"]]
        if d["transition_state"] is None:
            ts = None
            rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactants, products, transition_state=ts,
                       parameters=parameters)
        reaction.rate_calculator = rate_calculator
        return reaction


class IntermolecularReaction(Reaction):

    """
    A class to define intermolecular bond change as follows:
        Intermolecular formation / breakage of one bond
        A <-> B + C aka B + C <-> A
        Three entries with:
            comp(A) = comp(B) + comp(C)
            charge(A) = charge(B) + charge(C)
            removing one of the edges in A yields two disconnected subgraphs that are isomorphic to B and C

    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of two molecular entries
    """

    def __init__(self, reactant, product, transition_state=None, parameters=None):
        self.reactant = reactant[0]
        self.product0 = product[0]
        self.product1 = product[1]
        super().__init__([self.reactant], [self.product0, self.product1],
                         transition_state=transition_state,
                         parameters=parameters)

    @classmethod
    def generate(cls, entries):
        reactions = list()
        classes = dict()
        templates = list()
        for formula in entries:
            for Nbonds in entries[formula]:
                if Nbonds > 0:
                    for charge in entries[formula][Nbonds]:
                        if charge not in classes:
                            classes[charge] = dict()
                        for entry in entries[formula][Nbonds][charge]:
                            for edge in entry.edges:
                                bond = [(edge[0], edge[1])]
                                try:
                                    frags = entry.mol_graph.split_molecule_subgraphs(bond, allow_reverse=True)
                                    formula0 = frags[0].molecule.composition.alphabetical_formula
                                    Nbonds0 = len(frags[0].graph.edges())
                                    formula1 = frags[1].molecule.composition.alphabetical_formula
                                    Nbonds1 = len(frags[1].graph.edges())
                                    if formula0 in entries and formula1 in entries:
                                        if Nbonds0 in entries[formula0] and Nbonds1 in entries[formula1]:
                                            for charge0 in entries[formula0][Nbonds0]:
                                                for entry0 in entries[formula0][Nbonds0][charge0]:
                                                    if frags[0].isomorphic_to(entry0.mol_graph):
                                                        charge1 = charge - charge0
                                                        if charge1 in entries[formula1][Nbonds1]:
                                                            for entry1 in entries[formula1][Nbonds1][charge1]:
                                                                if frags[1].isomorphic_to(entry1.mol_graph):
                                                                    r = cls([entry], [entry0, entry1])
                                                                    indices = entry.mol_graph.extract_bond_environment([edge])
                                                                    subg = entry.graph.subgraph(list(indices)).copy().to_undirected()

                                                                    classes, templates = categorize(r, classes, templates, subg, charge)
                                                                    reactions.append(r)
                                                                    break
                                                        break
                                except MolGraphSplitError:
                                    pass

        return reactions, classes

    def reaction_type(self):
        rxn_type_A = "Molecular decomposition breaking one bond A -> B+C"
        rxn_type_B = "Molecular formation from one new bond A+B -> C"

        reaction_type = {"class": "IntermolecularReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self):
        entry = self.reactant
        entry0 = self.product0
        entry1 = self.product1

        if entry1.free_energy() is not None and entry0.free_energy() is not None and entry.free_energy() is not None:
            free_energy_A = entry0.free_energy() + entry1.free_energy() - entry.free_energy()
            free_energy_B = entry.free_energy() - entry0.free_energy() - entry1.free_energy()

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self):
        entry = self.reactant
        entry0 = self.product0
        entry1 = self.product1

        if entry1.energy is not None and entry0.energy is not None and entry.energy is not None:
            energy_A = entry0.energy + entry1.energy - entry.energy
            energy_B = entry.energy - entry0.energy - entry1.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate_constant(self):
        if isinstance(self.rate_calculator, ReactionRateCalculator):
            return {"k_A": self.rate_calculator.calculate_rate_constant(),
                    "k_B": self.rate_calculator.calculate_rate_constant(reverse=True)}
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            # No reference is set
            # Use barrierless reaction
            if self.rate_calculator.alpha == -1:
                rate_constant = dict()
                free_energy = self.free_energy()

                if free_energy["free_energy_A"] < 0:
                    rate_constant["k_A"] = k * 298.15 / h
                else:
                    rate_constant["k_A"] = k * 298.15 / h * np.exp(-1 * free_energy["free_energy_A"] * 96487 / (R * 298.15))

                if free_energy["free_energy_B"] < 0:
                    rate_constant["k_B"] = k * 298.15 / h
                else:
                    rate_constant["k_B"] = k * 298.15 / h * np.exp(-1 * free_energy["free_energy_B"] * 96487 / (R * 298.15))

                return rate_constant
            else:
                return {"k_A": self.rate_calculator.calculate_rate_constant(),
                        "k_B": self.rate_calculator.calculate_rate_constant(reverse=True)}

    def as_dict(self) -> dict:
        if self.transition_state is None:
            ts = None
        else:
            ts = self.transition_state.as_dict()

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "reactants": [r.as_dict() for r in self.reactants],
             "products": [p.as_dict() for p in self.products],
             "reactant": self.reactant.as_dict(),
             "product0": self.product0.as_dict(),
             "product1": self.product1.as_dict(),
             "transition_state": ts,
             "rate_calculator": self.rate_calculator.as_dict(),
             "parameters": self.parameters}

        return d

    @classmethod
    def from_dict(cls, d):
        reactants = [MoleculeEntry.from_dict(r) for r in d["reactants"]]
        products = [MoleculeEntry.from_dict(p) for p in d["products"]]
        if d["transition_state"] is None:
            ts = None
            rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactants, products, transition_state=ts,
                       parameters=parameters)
        reaction.rate_calculator = rate_calculator
        return reaction


class CoordinationBondChangeReaction(Reaction):

    """
    A class to define coordination bond change as follows:
        Simultaneous formation / breakage of multiple coordination bonds
        A + M <-> AM aka AM <-> A + M
        Three entries with:
            M = Li or Mg
            comp(AM) = comp(A) + comp(M)
            charge(AM) = charge(A) + charge(M)
            removing two M-containing edges in AM yields two disconnected subgraphs that are isomorphic to B and C

    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of two molecular entries
    """

    def __init__(self, reactant, product, transition_state=None, parameters=None):
        self.reactant = reactant[0]
        self.product0 = product[0]
        self.product1 = product[1]
        super().__init__([self.reactant],
                         [self.product0, self.product1],
                         transition_state=transition_state,
                         parameters=parameters)

    @classmethod
    def generate(cls, entries):
        reactions = list()
        M_entries = dict()
        classes = dict()
        templates = list()
        for formula in entries:
            if formula == "Li1" or formula == "Mg1":
                if formula not in M_entries:
                    M_entries[formula] = dict()
                for charge in entries[formula][0]:
                    assert (len(entries[formula][0][charge]) == 1)
                    M_entries[formula][charge] = entries[formula][0][charge][0]
        if M_entries != dict():
            for formula in entries:
                if "Li" in formula or "Mg" in formula:
                    for Nbonds in entries[formula]:
                        if Nbonds > 2:
                            for charge in entries[formula][Nbonds]:
                                if charge not in classes:
                                    classes[charge] = dict()
                                for entry in entries[formula][Nbonds][charge]:
                                    nosplit_M_bonds = list()
                                    for edge in entry.edges:
                                        if str(entry.molecule.sites[edge[0]].species) in M_entries or str(
                                                entry.molecule.sites[edge[1]].species) in M_entries:
                                            M_bond = (edge[0], edge[1])
                                            try:
                                                frags = entry.mol_graph.split_molecule_subgraphs([M_bond],
                                                                                                 allow_reverse=True)
                                            except MolGraphSplitError:
                                                nosplit_M_bonds.append(M_bond)
                                    bond_pairs = itertools.combinations(nosplit_M_bonds, 2)
                                    for bond_pair in bond_pairs:
                                        try:
                                            frags = entry.mol_graph.split_molecule_subgraphs(bond_pair,
                                                                                             allow_reverse=True)
                                            M_ind = None
                                            M_formula = None
                                            for ii, frag in enumerate(frags):
                                                frag_formula = frag.molecule.composition.alphabetical_formula
                                                if frag_formula in M_entries:
                                                    M_ind = ii
                                                    M_formula = frag_formula
                                                    break
                                            if M_ind != None:
                                                for ii, frag in enumerate(frags):
                                                    if ii != M_ind:
                                                        nonM_formula = frag.molecule.composition.alphabetical_formula
                                                        nonM_Nbonds = len(frag.graph.edges())
                                                        if nonM_formula in entries:
                                                            if nonM_Nbonds in entries[nonM_formula]:
                                                                for nonM_charge in entries[nonM_formula][nonM_Nbonds]:
                                                                    M_charge = entry.charge - nonM_charge
                                                                    if M_charge in M_entries[M_formula]:
                                                                        for nonM_entry in \
                                                                                entries[nonM_formula][nonM_Nbonds][
                                                                                    nonM_charge]:
                                                                            if frag.isomorphic_to(nonM_entry.mol_graph):
                                                                                r = cls([entry],[nonM_entry,
                                                                                                 M_entries[M_formula][M_charge]])

                                                                                indices = entry.mol_graph.extract_bond_environment([edge])
                                                                                subg = entry.graph.subgraph(list(indices)).copy().to_undirected()

                                                                                classes, templates = categorize(r, classes, templates, subg, charge)

                                                                                reactions.append(r)
                                                                                break
                                        except MolGraphSplitError:
                                            pass
        return reactions, classes

    def reaction_type(self):
        rxn_type_A = "Coordination bond breaking AM -> A+M"
        rxn_type_B = "Coordination bond forming A+M -> AM"

        reaction_type = {"class": "CoordinationBondChangeReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self):
        entry = self.reactant
        entry0 = self.product0
        entry1 = self.product1

        if entry1.free_energy() is not None and entry0.free_energy() is not None and entry.free_energy() is not None:
            free_energy_A = entry0.free_energy() + entry1.free_energy() - entry.free_energy()
            free_energy_B = entry.free_energy() - entry0.free_energy() - entry1.free_energy()

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self):
        entry = self.reactant
        entry0 = self.product0
        entry1 = self.product1

        if entry1.energy is not None and entry0.energy is not None and entry.energy is not None:
            energy_A = entry0.energy + entry1.energy - entry.energy
            energy_B = entry.energy - entry0.energy - entry1.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate_constant(self):
        if isinstance(self.rate_calculator, ReactionRateCalculator):
            return {"k_A": self.rate_calculator.calculate_rate_constant(),
                    "k_B": self.rate_calculator.calculate_rate_constant(reverse=True)}
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            # No reference is set
            # Use barrierless reaction
            if self.rate_calculator.alpha == -1:
                rate_constant = dict()
                free_energy = self.free_energy()

                if free_energy["free_energy_A"] < 0:
                    rate_constant["k_A"] = k * 298.15 / h
                else:
                    rate_constant["k_A"] = k * 298.15 / h * np.exp(-1 * free_energy["free_energy_A"] * 96487 / (R * 298.15))

                if free_energy["free_energy_B"] < 0:
                    rate_constant["k_B"] = k * 298.15 / h
                else:
                    rate_constant["k_B"] = k * 298.15 / h * np.exp(-1 * free_energy["free_energy_B"] * 96487 / (R * 298.15))

                return rate_constant
            else:
                return {"k_A": self.rate_calculator.calculate_rate_constant(),
                        "k_B": self.rate_calculator.calculate_rate_constant(reverse=True)}

    def as_dict(self) -> dict:
        if self.transition_state is None:
            ts = None
        else:
            ts = self.transition_state.as_dict()

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "reactants": [r.as_dict() for r in self.reactants],
             "products": [p.as_dict() for p in self.products],
             "reactant": self.reactant.as_dict(),
             "product0": self.product0.as_dict(),
             "product1": self.product1.as_dict(),
             "transition_state": ts,
             "rate_calculator": self.rate_calculator.as_dict(),
             "parameters": self.parameters}

        return d

    @classmethod
    def from_dict(cls, d):
        reactants = [MoleculeEntry.from_dict(r) for r in d["reactants"]]
        products = [MoleculeEntry.from_dict(p) for p in d["products"]]
        if d["transition_state"] is None:
            ts = None
            rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactants, products, transition_state=ts,
                       parameters=parameters)
        reaction.rate_calculator = rate_calculator
        return reaction


def get_node_names_1_2(reaction):
    """
    Give standard naming for reaction nodes.
    Reaction must be of type 1 reactant -> 2 products.

    Args:
        reaction: any Reaction class object, ex. IntermolecularReaction
    Returns:
        names: list of str
    """

    entry = reaction.reactant
    entry0 = reaction.product0
    entry1 = reaction.product1

    if entry0.parameters["ind"] <= entry1.parameters["ind"]:
        two_mol_name = str(entry0.parameters["ind"]) + "+" + str(entry1.parameters["ind"])
    else:
        two_mol_name = str(entry1.parameters["ind"]) + "+" + str(entry0.parameters["ind"])

    two_mol_name0 = str(entry0.parameters["ind"]) + "+PR_" + str(entry1.parameters["ind"])
    two_mol_name1 = str(entry1.parameters["ind"]) + "+PR_" + str(entry0.parameters["ind"])
    node_name_A = str(entry.parameters["ind"]) + "," + two_mol_name
    node_name_B0 = two_mol_name0 + "," + str(entry.parameters["ind"])
    node_name_B1 = two_mol_name1 + "," + str(entry.parameters["ind"])

    return {"node_name_A": node_name_A, "node_name_B0": node_name_B0,
            "node_name_B1": node_name_B1}


def get_node_names_1_1(reaction):
    """
    Give standard naming for reaction nodes.
    Reaction must be of type 1 reactant -> 1 product

    Args:
        reaction: any Reaction class object, ex. IntramolecularSingleBondChangeReaction
    Returns:
        names: list of str
    """
    entry0 = reaction.reactant
    entry1 = reaction.product
    node_name_A = str(entry0.parameters["ind"]) + "," + str(entry1.parameters["ind"])
    node_name_B = str(entry1.parameters["ind"]) + "," + str(entry0.parameters["ind"])

    return {"node_name_A": node_name_A, "node_name_B": node_name_B}


def graph_rep_1_2(reaction) -> nx.DiGraph:
    """
    A method to convert a reaction type object into graph representation.
    Reaction must be of type 1 reactant -> 2 products

    Args:
       reaction (any of the Reaction class object, ex. IntermolecularReaction)

    Returns:
        graph: nx.DiGraph representing the connections between the reactants,
            products, and this reaction.
    """

    entry = reaction.reactant
    entry0 = reaction.product0
    entry1 = reaction.product1
    graph = nx.DiGraph()

    if entry0.parameters["ind"] <= entry1.parameters["ind"]:
        two_mol_name_entry_ids = str(entry0.entry_id) + "+" + str(entry1.entry_id)
    else:
        two_mol_name_entry_ids = str(entry1.entry_id) + "+" + str(entry0.entry_id)

    names = get_node_names_1_2(reaction)
    node_name_A = names['node_name_A']
    node_name_B0 = names['node_name_B0']
    node_name_B1 = names['node_name_B1']

    two_mol_entry_ids0 = str(entry0.entry_id) + "+PR_" + str(entry1.entry_id)
    two_mol_entry_ids1 = str(entry1.entry_id) + "+PR_" + str(entry0.entry_id)
    entry_ids_name_A = str(entry.entry_id) + "," + two_mol_name_entry_ids
    entry_ids_name_B0 = two_mol_entry_ids0 + "," + str(entry.entry_id)
    entry_ids_name_B1 = two_mol_entry_ids1 + "," + str(entry.entry_id)

    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]

    graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                   entry_ids=entry_ids_name_A)

    graph.add_edge(entry.parameters["ind"],
                   node_name_A,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_A,
                   entry0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_A,
                   entry1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B0)
    graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B1)

    graph.add_edge(node_name_B0,
                   entry.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_B1,
                   entry.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_edge(entry0.parameters["ind"],
                   node_name_B0,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0
                   )
    graph.add_edge(entry1.parameters["ind"],
                   node_name_B1,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0)
    return graph


def graph_rep_1_1(reaction) -> nx.DiGraph:

    """
    A method to convert a reaction type object into graph representation. Reaction much be of type 1 reactant -> 1
    product

    Args:
       reactant (any of the reaction class object, ex. RedoxReaction, IntramolSingleBondChangeReaction):
    """

    entry0 = reaction.reactant
    entry1 = reaction.product
    graph = nx.DiGraph()
    node_name_A = get_node_names_1_1(reaction)["node_name_A"]
    node_name_B = get_node_names_1_1(reaction)["node_name_B"]
    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]
    entry_ids_A = str(entry0.entry_id) + "," + str(entry1.entry_id)
    entry_ids_B = str(entry1.entry_id) + "," + str(entry0.entry_id)

    graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                   entry_ids=entry_ids_A)
    graph.add_edge(entry0.parameters["ind"],
                   node_name_A,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   weight=1.0)
    graph.add_edge(node_name_A,
                   entry1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0)
    graph.add_node(node_name_B, rxn_type=rxn_type_B, bipartite=1,energy_B=energy_B,free_energy=free_energy_B,
                   entry_ids=entry_ids_B)
    graph.add_edge(entry1.parameters["ind"],
                   node_name_B,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0)
    graph.add_edge(node_name_B,
                   entry0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0)

    return graph


class ReactionNetwork:
    """
        Class to build a reaction network from molecules and reactions.
    """

    def __init__(self, electron_free_energy, entries_dict, entries_list,
                 graph, reactions, classes, mapping, PR_record, min_cost,
                 num_starts):
        """
        :param electron_free_energy: Electron free energy (in eV)
        :param entries_dict: dict of dicts of dicts of lists (d[formula][bonds][charge])
        :param entries_list: list of unique entries in entries_dict
        :param graph: nx.DiGraph representing connections in the network
        :param reactions: list of Reaction objects
        :param classes: dict containing reaction families
        :param mapping: dict linking rxn node names to rxn node indices
            (along with information about directionality)
        :param PR_record: dict containing reaction prerequisites
        :param min_cost: dict containing costs of entries in the network
        :param num_starts: <-- What DOES this represent?
        """

        self.electron_free_energy = electron_free_energy

        self.entries = entries_dict
        self.entries_list = entries_list

        self.graph = graph
        self.PR_record = PR_record
        self.reactions = reactions
        self.classes = classes
        self.rxn_node_to_rxn_ind = mapping

        self.min_cost = min_cost
        self.num_starts = num_starts

    @classmethod
    def from_input_entries(cls, input_entries, electron_free_energy=-2.15):
        """
        Generate a ReactionNetwork from a set of MoleculeEntries

        :param input_entries: list of MoleculeEntries which will make up the
            network
        :param electron_free_energy: float representing the Gibbs free energy
            required to add an electron
        :return:
        """

        entries = dict()
        entries_list = list()

        print(len(input_entries), "input entries")

        connected_entries = list()
        for entry in input_entries:
            if len(entry.molecule) > 1:
                if nx.is_weakly_connected(entry.graph):
                    connected_entries.append(entry)
            else:
                connected_entries.append(entry)
        print(len(connected_entries), "connected entries")

        get_formula = lambda x: x.formula
        get_Nbonds = lambda x: x.Nbonds
        get_charge = lambda x: x.charge
        get_free_energy = lambda x: x.free_energy()

        sorted_entries_0 = sorted(connected_entries, key=get_formula)
        for k1, g1 in itertools.groupby(sorted_entries_0, get_formula):
            sorted_entries_1 = sorted(list(g1), key=get_Nbonds)
            entries[k1] = dict()

            for k2, g2 in itertools.groupby(sorted_entries_1, get_Nbonds):
                sorted_entries_2 = sorted(list(g2), key=get_charge)
                entries[k1][k2] = dict()
                for k3, g3 in itertools.groupby(sorted_entries_2, get_charge):
                    entries_3 = list(g3)
                    sorted_entries_3 = sorted(entries_3, key=get_free_energy)
                    if len(sorted_entries_3) > 1:
                        unique = list()
                        for entry in sorted_entries_3:
                            isomorphic_found = False
                            for ii, Uentry in enumerate(unique):
                                if entry.mol_graph.isomorphic_to(Uentry.mol_graph):
                                    isomorphic_found = True
                                    if entry.free_energy() is not None and Uentry.free_energy() is not None:
                                        if entry.free_energy() < Uentry.free_energy():
                                            unique[ii] = entry
                                    elif entry.free_energy is not None:
                                        unique[ii] = entry
                                    elif entry.energy < Uentry.energy:
                                        unique[ii] = entry
                                    break
                            if not isomorphic_found:
                                unique.append(entry)
                        entries[k1][k2][k3] = unique
                    else:
                        entries[k1][k2][k3] = sorted_entries_3
                    for entry in entries[k1][k2][k3]:
                        entries_list.append(entry)

        print(len(entries_list), "unique entries")
        for ii, entry in enumerate(entries_list):
            if "ind" in entry.parameters.keys():
                pass
            else:
                entry.parameters["ind"] = ii

        entries_list = sorted(entries_list, key=lambda x: x.parameters["ind"])

        graph = nx.DiGraph()

        network = cls(electron_free_energy, entries, entries_list, graph,
                      list(), dict(), dict(), None, dict(), None)

        return network

    @property
    def reaction_labels(self):
        return {n for n, d in self.graph.nodes(data=True) if d["bipartite"] == 1}

    @staticmethod
    def softplus(free_energy):
        return np.log(1 + (273.0 / 500.0) * np.exp(free_energy))

    @staticmethod
    def exponent(free_energy):
        return np.exp(free_energy)

    def build(self, reaction_types=None):
        """
        Build a ReactionNetwork based on certain reaction types.

        Args:
            reaction_types (set of str): Class names for different reaction
                classes
            transition_states (list of MoleculeEntry objects): transition states
                to be associated with reactions for rate calculations
        :return:
        """

        if reaction_types is None:
            reaction_types = {"RedoxReaction",
                              "IntramolSingleBondChangeReaction",
                              "IntermolecularReaction",
                              "CoordinationBondChangeReaction"}

        reaction_classes = {s: load_class(str(self.__module__)+"."+s) for s in reaction_types}

        all_reactions = list()
        classes = dict()
        # Generate reactions separately by reaction type
        for rtype, rclass in reaction_classes.items():
            reactions, classdict = rclass.generate(self.entries)
            self.classes[rtype] = classes
            all_reactions.append(reactions)

        # Compile reactions
        self.reactions = [i for i in self.reactions if i]
        self.reactions = list(itertools.chain.from_iterable(all_reactions))
        self.graph.add_nodes_from(range(len(self.entries_list)), bipartite=0)

        # Fill out graph with reaction nodes
        # Also map reaction node names to reaction ids, entry dicts, and
        # direction (forwards = False, reverse = True)
        mapping = dict()

        for ii, r in enumerate(self.reactions):
            r.parameters["ind"] = ii
            if r.reaction_type()["class"] == "RedoxReaction":
                r.electron_free_energy = self.electron_free_energy
                self.add(graph_rep_1_1(r))
                names = get_node_names_1_1(r)
            elif r.reaction_type()["class"] == "IntramolSingleBondChangeReaction":
                self.add(graph_rep_1_1(r))
                names = get_node_names_1_1(r)
            else:
                self.add(graph_rep_1_2(r))
                names = get_node_names_1_2(r)

            for namelabel, name in names.items():
                if "A" in namelabel:
                    mapping[name] = (ii, r.reactant_ids, r.product_ids, False)
                else:
                    mapping[name] = (ii, r.reactant_ids, r.product_ids, True)

        self.rxn_node_to_rxn_ind = mapping

        # Add index parameters to reactions in self.classes as well as reactions
        # This is inefficient, but helps significantly with later functions
        for reaction in self.reactions:
            for layer1, class1 in self.classes[r.reaction_type()["class"]].items():
                for layer2, class2 in class1.items():
                    for rxn in class2:
                        # Reactions identical - link by index
                        if reaction.reactant_ids == rxn.reacant_ids and reaction.product_ids == rxn.product_ids:
                            rxn.parameters["ind"] = reaction.parameters["ind"]

        return self.graph

    def add(self, graph_representation):
        """
        Add molecule and reaction nodes to the graph representation.

        Args:
            graph_representation (nx.DiGraph): new graph to be added to the
                network
        Returns:
            None
        """
        self.graph.add_nodes_from(graph_representation.nodes(data=True))
        self.graph.add_edges_from(graph_representation.edges(data=True))

    def associate_transition_states(self, ts_sets):
        """
        Add transition states to Reactions in the ReactionNetwork, thereby
            allowing for rate calculation.

        Args:
            ts_sets: list of dicts {"reactants": [MoleculeEntry],
                "products": [MoleculeEntry],
                "ts": MoleculeEntry}
        Returns:
            None
        """

        # Iterate by reaction class
        for rtype, bigclasses in self.classes.items():
            for bigclass, smallclasses in bigclasses.items():
                for smallclass, reactions in smallclasses.items():
                    reference_ts = None
                    # Try to associate any TS with any reaction in the class
                    # Update reaction list as you go
                    for reaction in reactions:
                        ind = reaction.parameters.get("ind", None)
                        if reaction.transition_state is None:
                            for ts_set in ts_sets:
                                ts_rct_ids = [r.entry_id for r in ts_set["reactants"]]
                                rct_ids = [r.entry_id for r in reaction.reactants]
                                ts_pro_ids = [p.entry_id for p in ts_set["products"]]
                                pro_ids = [p.entry_id for p in reaction.products]
                                if sorted(rct_ids) == sorted(ts_rct_ids) and sorted(pro_ids) == sorted(ts_pro_ids):
                                    reaction.update_calculator(transition_state=ts_set["ts"])
                                    if ind is not None:
                                        # Make sure labeling is correct
                                        if self.reactions[ind].parameters["ind"] == ind:
                                            self.reactions[ind].update_calculator(transition_state=ts_set["ts"])

                                    # Supply reference reaction data
                                    if reference_ts is None:
                                        calc = reaction.rate_calculator
                                        reference_ts = {"delta_ea": calc.calculate_act_energy(),
                                                        "delta_ha": calc.calculate_act_enthalpy(),
                                                        "delta_sa": calc.calculate_act_entropy(),
                                                        "delta_e": calc.net_energy,
                                                        "delta_h": calc.net_enthalpy,
                                                        "delta_s": calc.net_entropy}
                                    break

                    # Once association has been made
                    # If a reference has been found, update all others using
                    # the reference
                    # Again, update reaction list as you go
                    if reference_ts is not None:
                        for reaction in reactions:
                            ind = reaction.parameters.get("ind", None)
                            if reaction.transition_state is None:
                                reaction.update_calculator(reference=reference_ts)

                                if ind is not None:
                                    if self.reactions[ind].parameters["ind"] == ind:
                                        self.reactions[ind].update_calculator(reference=reference_ts)

    def get_reactions_by_entries(self, entries):
        """
        Find all reactions involving only the specified molecules.

        Args:
            entries (list of MoleculeEntries)
        Returns:
            list of Reaction objects
        """
        reactions = set()
        mol_ids = [e.entry_id for e in entries]
        for entry in entries:
            index = entry.parameters["ind"]

            neighbors = self.graph[index]
            for neighbor, _ in neighbors.items():
                mapping_entry = self.rxn_node_to_rxn_ind[neighbor]
                all_included = True
                for rct_id in mapping_entry[1]:
                    if rct_id not in mol_ids:
                        all_included = False
                for pro_id in mapping_entry[2]:
                    if pro_id not in mol_ids:
                        all_included = False
                if all_included:
                    reactions.add(mapping_entry[0])

        return [self.reactions[ii] for ii in reactions]

    def as_dict(self) -> dict:
        entries = dict()
        for formula in self.entries.keys():
            entries[formula] = dict()
            for bonds in self.entries[formula].keys():
                entries[formula][bonds] = dict()
                for charge in self.entries[formula][bonds].keys():
                    entries[formula][bonds][charge] = list()
                    for entry in self.entries[formula][bonds][charge]:
                        entries[formula][bonds][charge].append(entry.as_dict())

        entries_list = [e.as_dict() for e in self.entries_list]

        reactions = [r.as_dict() for r in self.reactions]

        classes = dict()
        for category in self.classes.keys():
            classes[category] = dict()
            for charge in self.classes[category].keys():
                classes[category][charge] = dict()
                for label in self.classes[category][charge].keys():
                    classes[category][charge][label] = list()
                    for reaction in self.classes[category][charge][label]:
                        classes[category][charge][label].append(reaction.as_dict())

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "entries_dict": entries,
             "entries_list": entries_list,
             "reactions": reactions,
             "classes": classes,
             "mapping": self.rxn_node_to_rxn_ind,
             "electron_free_energy": self.electron_free_energy,
             "graph": json_graph.adjacency_data(self.graph),
             "PR_record": self.PR_record,
             "min_cost": self.min_cost,
             "num_starts": self.num_starts}

        return d

    @classmethod
    def from_dict(cls, d):

        entries = dict()
        d_entries = d["entries_dict"]
        for formula in d_entries.keys():
            entries[formula] = dict()
            for bonds in d_entries[formula].keys():
                entries[formula][bonds] = dict()
                for charge in d_entries[formula][bonds].keys():
                    entries[formula][bonds][charge] = list()
                    for entry in d_entries[formula][bonds][charge]:
                        entries[formula][bonds][charge].append(MoleculeEntry.from_dict(entry))

        entries_list = [MoleculeEntry.from_dict(e) for e in d["entries_list"]]

        reactions = list()
        for reaction in d["reactions"]:
            rclass = load_class(str(cls.__module__)+"."+ reaction["@class"])
            reactions.append(rclass.from_dict(reaction))

        classes = dict()
        for category in d["classes"].keys():
            classes[category] = dict()
            for charge in d["classes"][category].keys():
                classes[category][charge] = dict()
                for label in d["classes"][category][charge].keys():
                    classes[category][charge][label] = list()
                    for reaction in d["classes"][category][charge][label]:
                        rclass = load_class(str(cls.__module__) + "." + reaction["@class"])
                        classes[category][charge][label].append(rclass.from_dict(reaction))

        graph = json_graph.adjacency_graph(d["graph"], directed=True)

        return cls(d["electron_free_energy"], entries, entries_list, graph,
                   reactions, classes, d["mapping"], d["PR_record"],
                   d["min_cost"], d["num_starts"])
