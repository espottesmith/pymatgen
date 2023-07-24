import os

import pytest

from pymatgen.core import SETTINGS, Structure
from pymatgen.core.structure import Molecule
from pymatgen.io.orca.inputs import ORCAInput
from pymatgen.io.orca.sets import ORCASet


__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2023, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"


module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture(scope="session")
def test_dir():
    if "PMG_TEST_FILES_DIR" in os.environ:
        test_dir = os.environ["PMG_TEST_FILES_DIR"]
    else:
        module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        test_dir = os.path.join(module_dir, "..", "..", "..", "test_files")

    return test_dir

def test_read_write(test_dir):
    os.mkdir("tmp")
    try:
        molecule = Molecule(["O", "H", "H"])

        inp = ORCAInput(

        )
    finally:
        os.rmdir("tmp")