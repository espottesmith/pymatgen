import pytest

from monty.serialization import loadfn, dumpfn

from pymatgen.core.structure import Molecule
from pymatgen.io.orca.inputs import ORCAInput

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2023, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__credits__ = "Sam Blau"

logger = logging.getLogger(__name__)


def test_molecule_template():
     mol = Molecule(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.91]], charge=-1)
     template = ORCAInput.molecule_template(mol)
     assert template == """* xyz -1 1
 O      0.0000000000      0.0000000000      0.0000000000
 H      0.0000000000      0.0000000000      0.9100000000
*"""

def test_template():
    block = {"array": [1, 2, 3],
             "string": "str",
             "integer": 1,
             "float": 1.337}

def test_find_blocks():
    pass

def test_read_molecule():
    pass

def test_read_simple_input():
    pass

def test_read_block():
    pass