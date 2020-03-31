from abc import ABCMeta, abstractproperty, abstractmethod, abstractclassmethod
from abc import ABC, abstractmethod
from gunicorn.util import load_class

import logging
import copy
import itertools
import heapq

import numpy as np
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
                                               BEPRateCalculator,
                                               ExpandedBEPRateCalculator)


def categorize(reaction, classes, templates, environment, charge):
    """
    Given reactants, products, and a local bonding environment, place a
        reaction into a "bucket".

    Note: This is not designed for redox reactions

    :param reaction: Reaction object
    :param classes: dict of dicts representing families of reactions
    :param environment: a nx.Graph object representing a submolecule that
        defines the type of reaction
    :param templates: list of nx.Graph objects that define other classes
    :param charge: int representing the charge of the reaction
    :return:
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

    def __init__(self, reactants, products, transition_state=None):
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
        self.entry_ids = {e.entry_id for e in self.reactants}

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

    def __init__(self, reactant, product, transition_state=None):
        if len(reactant) != 1 or len(product) != 1:
            raise RuntimeError("One electron redox requires two lists that each contain one entry!")
        self.reactant = reactant[0]
        self.product = product[0]
        self.electron_free_energy = None
        super().__init__([self.reactant], [self.product],
                         transition_state=transition_state)

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

    def __init__(self, reactant, product, transition_state=None):
        if len(reactant) != 1 or len(product) != 1:
            raise RuntimeError("Intramolecular single bond change requires two lists that each contain one entry!")
        self.reactant = reactant[0]
        self.product = product[0]
        super().__init__([self.reactant], [self.product], transition_state=transition_state)

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
        pass


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

    def __init__(self, reactant, product, transition_state=None):
        self.reactant = reactant[0]
        self.product0 = product[0]
        self.product1 = product[1]
        super().__init__([self.reactant], [self.product0, self.product1], transition_state=transition_state)

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
        pass


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

    def __init__(self, reactant, product):
        self.reactant = reactant[0]
        self.product0 = product[0]
        self.product1 = product[1]
        super().__init__([self.reactant],
                         [self.product0, self.product1])

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
        pass


def graph_rep_1_2(reaction) -> nx.DiGraph:

    """
    A method to convert a reaction type object into graph representation.
    Reaction much be of type 1 reactant -> 2 products

    Args:
       reaction (any of the Reaction class object, ex. IntermolecularReaction)
    """

    entry = reaction.reactant
    entry0 = reaction.product0
    entry1 = reaction.product1
    graph = nx.DiGraph()

    if entry0.parameters["ind"] <= entry1.parameters["ind"]:
        two_mol_name = str(entry0.parameters["ind"]) + "+" + str(entry1.parameters["ind"])
        two_mol_name_entry_ids = str(entry0.entry_id) + "+" + str(entry1.entry_id)
    else:
        two_mol_name = str(entry1.parameters["ind"]) + "+" + str(entry0.parameters["ind"])
        two_mol_name_entry_ids = str(entry1.entry_id) + "+" + str(entry0.entry_id)

    two_mol_name0 = str(entry0.parameters["ind"]) + "+PR_" + str(entry1.parameters["ind"])
    two_mol_name1 = str(entry1.parameters["ind"]) + "+PR_" + str(entry0.parameters["ind"])
    node_name_A = str(entry.parameters["ind"]) + "," + two_mol_name
    node_name_B0 = two_mol_name0 + "," + str(entry.parameters["ind"])
    node_name_B1 = two_mol_name1 + "," + str(entry.parameters["ind"])

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
    node_name_A = str(entry0.parameters["ind"]) + "," + str(entry1.parameters["ind"])
    node_name_B = str(entry1.parameters["ind"]) + "," + str(entry0.parameters["ind"])
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

    def __init__(self, electron_free_energy, entries_dict, entries_list, classes,
                 graph, reactions, PR_record, min_cost, num_starts):
        """

        :param electron_free_energy: Electron free energy (in eV)
        :param entries_dict: dict of dicts of dicts of lists (d[formula][bonds][charge])
        :param entries_list: list of unique entries in entries_dict
        :param classes: dict containing reaction families
        :param graph: nx.DiGraph representing connections in the network
        :param PR_record: dict containing reaction prerequisites
        :param reactions: list of Reaction objects
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

        network = cls(electron_free_energy, entries, entries_list, dict(),
                      graph, list(), None, dict(), None)

        return network

    @staticmethod
    def softplus(free_energy):
        return np.log(1 + (273.0 / 500.0) * np.exp(free_energy))

    @staticmethod
    def exponent(free_energy):
        return np.exp(free_energy)

    def build(self, reaction_types=None, transition_states=None):
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
        for rtype, rclass in reaction_classes.items():
            reactions, classes = rclass.generate(self.entries,
                                                 transition_states=transition_states)
            self.classes[rtype] = classes
            all_reactions.append(reactions)
        self.reactions = [i for i in self.reactions if i]
        self.reactions = list(itertools.chain.from_iterable(all_reactions))
        self.graph.add_nodes_from(range(len(self.entries_list)), bipartite=0)
        for r in self.reactions:
            if r.reaction_type()["class"] == "RedoxReaction":
                r.electron_free_energy = self.electron_free_energy
                self.add(graph_rep_1_1(r))
            elif r.reaction_type()["class"] == "IntramolSingleBondChangeReaction":
                self.add(graph_rep_1_1(r))
            else:
                self.add(graph_rep_1_2(r))

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

    # def as_dict(self):
    #     entries = dict()
    #     for formula in self.entries.keys():
    #         entries[formula] = dict()
    #         for bonds in self.entries[formula].keys():
    #             entries[formula][bonds] = dict()
    #             for charge in self.entries[formula][bonds].keys():
    #                 entries[formula][bonds][charge] = list()
    #                 for entry in self.entries[formula][bonds][charge]:
    #                     entries[formula][bonds][charge].append(entry.as_dict())
    #
    #     entries_list = [e.as_dict() for e in self.entries_list]
    #
    #     reactions = [r.as_dict() for r in self.reactions]
    #
    #     classes = dict()
    #     for category in self.buckets.keys():
    #         classes[category] = dict()
    #         for charge in self.buckets[category].keys():
    #             classes[category][charge] = dict()
    #             for label in self.buckets[category][charge].keys():
    #                 classes[category][charge][label] = list()
    #                 for reaction in self.buckets[category][charge][label]:
    #                     classes[category][charge][label].append(reaction.as_dict())
    #
    #     d = {"@module": self.__class__.__module__,
    #          "@class": self.__class__.__name__,
    #          "entries_dict": entries,
    #          "entries_list": entries_list,
    #          "reactions": reactions,
    #          "classes": classes,
    #          "electron_free_energy": self.electron_free_energy,
    #          "graph": json_graph.adjacency_data(self.graph),
    #          "PR_record": self.PR_record,
    #          "min_cost": self.min_cost,
    #          "num_starts": self.num_starts,
    #          "local_order": self.local_order,
    #          "consider_charge_categories": self.consider_charge_categories}
    #
    #     return d
    #
    # @classmethod
    # def from_dict(cls, d):
    #
    #     entries = dict()
    #     d_entries = d["entries_dict"]
    #     for formula in d_entries.keys():
    #         entries[formula] = dict()
    #         for bonds in d_entries[formula].keys():
    #             entries[formula][bonds] = dict()
    #             for charge in d_entries[formula][bonds].keys():
    #                 entries[formula][bonds][charge] = list()
    #                 for entry in d_entries[formula][bonds][charge]:
    #                     entries[formula][bonds][charge].append(MoleculeEntry.from_dict(entry))
    #
    #     entries_list = [MoleculeEntry.from_dict(e) for e in d["entries_list"]]
    #
    #     for reaction in d["reactions"]:
    #         rclass =
    #
    #     classes = dict()
    #     for category in d["buckets"].keys():
    #         classes[category] = dict()
    #         if d["consider_charge_categories"]:
    #             for charge in d["buckets"][category].keys():
    #                 classes[category][charge] = dict()
    #                 for label in d["buckets"][category][charge].keys():
    #                     classes[category][charge][label] = list()
    #                     for reaction in d["buckets"][category][charge][label]:
    #                         classes[category][charge][label].append()
    #         else:
    #             for label in d["buckets"][category].keys():
    #                 buckets[category][label] = list()
    #                 for reaction in d["buckets"][category][label]:
    #                     rcts = [MoleculeEntry.from_dict(e) for e in reaction[0]]
    #                     pros = [MoleculeEntry.from_dict(e) for e in reaction[1]]
    #                     buckets[category][label].append((rcts, pros))
    #
    #     graph = json_graph.adjacency_graph(d["graph"], directed=True)
    #
    #     return cls(d["electron_free_energy"], entries, entries_list, buckets,
    #                graph, d["PR_record"], d["min_cost"], d["num_starts"])