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
    template = ORCAInput.template(block, "example")
    assert template == """%example
  array 1,2,3
  string str
  integer 1
  float 1.337
end
"""

def test_find_blocks():
    test_input = """! NORI CPCM Opt Freq ExtremeSCF

* xyz -1 1
 O      0.0000000000      0.0000000000      0.0000000000
 H      0.0000000000      0.0000000000      0.9100000000
*

%basis
  basis def2-SVPD
end

%cpcm
  smd True
  smdsolvent water
end

%geom
  maxiter 100
end

%method
  method wB97M-V
end

%pal
  nprocs 2
end"""

    blocks = ORCAInput.find_blocks(test_input)
    assert blocks == ["basis", "cpcm", "geom", "method", "pal"]

def test_read_molecule():
    mol_block = """* xyz -1 1
 O      0.0000000000      0.0000000000      0.0000000000
 H      0.0000000000      0.0000000000      0.9100000000
*"""

    mol = ORCAInput.read_molecule(mol_block)
    reference = Molecule(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.91]], charge=-1.0)
    assert mol == reference

def test_read_simple_input():
    si_example = "! NORI CPCM Opt Freq ExtremeSCF"

    parsed = ORCAInput.read_simple_input(si_example)
    assert parsed == ["NORI", "CPCM", "Opt", "Freq", "ExtremeSCF"]

def test_read_block():
    test_input = """! NORI CPCM Opt Freq ExtremeSCF

* xyz -1 1
 O      0.0000000000      0.0000000000      0.0000000000
 H      0.0000000000      0.0000000000      0.9100000000
*

%basis
  basis def2-SVPD
end

%cpcm
  smd True
  smdsolvent water
end

%geom
  maxiter 100
end

%method
  method wB97M-V
end

%pal
  nprocs 2
end"""

    cpcm_block = ORCAInput.read_block(test_input, "cpcm")
    
    assert cpcm_block == {"smd": "True", "smdsolvent": "water"}