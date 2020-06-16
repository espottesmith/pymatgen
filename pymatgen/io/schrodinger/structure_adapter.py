import copy

import numpy as np

from schrodinger.structure import StructureReader, Structure

from pymatgen.core.structure import Molecule
from pymatgen.core.sites import Site


def schrodinger_struct_to_molecule(structure: Structure):
    """
    Convert a Structure object from Schrodinger to a pymatgen Molecule object.

    Args:
        structure (schrodinger.structure.Structure object): Structure to be
            converted

    Returns:
        mol: pymatgen.core.structure.Molecule object
    """

    formal_charge = structure.formal_charge

    elements = list()
    positions = list()
    for molecule in structure.molecule:
        for atom in molecule.atom:
            elements.append(atom.element)
            positions.append(atom.xyz)

    mol = Molecule(elements, positions)
    mol.set_charge_and_spin(charge=formal_charge)

    return mol


def molecule_to_schrodinger_struct(molecule: Molecule):
    """
    Convert a pymatgen Molecule object to a Schrodinger Structure object

    Args:
        molecule (pymatgen.core.structure.Molecule): Molecule to be converted

    Returns:
        struct: schrodinger.structure.Structure object
    """

