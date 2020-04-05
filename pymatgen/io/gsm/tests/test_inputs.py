# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import logging
import os
import unittest

from monty.serialization import loadfn

from pymatgen import Molecule
from pymatgen.util.testing import PymatgenTest
from pymatgen.io.gsm.inputs import QCTemplate, GSMIsomerInput

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"
__email__ = "espottesmith@gmail.com"

logger = logging.getLogger(__name__)

test_dir = os.path.join(os.path.dirname(__file__), "..",
                        "..", "..", "..", "test_files", "gsm")


class TestQCTemplate(PymatgenTest):

    def test_create(self):
        # Test without basis
        rem = {"xc_grid": 3,
               "max_scf_cycles": 200,
               "scf_algorithm": "diis",
               "thresh": 14}

        with self.assertRaises(ValueError):
            test = QCTemplate(rem)

        # Test without method
        rem["basis"] = "6-311++g(d,p)"

        with self.assertRaises(ValueError):
            test = QCTemplate(rem)

        # Test with job type other than force
        rem["method"] = "wb97xd"
        rem["job_type"] = "fsm"

        test = QCTemplate(rem)

        self.assertEqual(test.rem["job_type"], "force")

        # Test correct input
        rem["job_type"] = "force"
        opt = {"FIXED": ["2 XY"]}
        smx = {"solvent": "thf"}

        test = QCTemplate(rem, opt=opt, smx=smx)
        self.assertDictEqual(test.rem, rem)
        self.assertDictEqual(test.opt, opt)
        self.assertDictEqual(test.smx, smx)

    def test_str(self):
        rem = {"basis": "6-31*",
               "method": "b3lyp",
               "job_type": "force"}
        pcm = {"theory": "cpcm"}
        solvent = {"dielectric": 80.4}

        test = QCTemplate(rem, pcm=pcm, solvent=solvent)

        self.assertEqual(str(test),
                         '$rem\n   basis = 6-31*\n   method = b3lyp\n   job_type = force\n$end\n\n$pcm\n   theory cpcm\n$end\n\n$solvent\n   dielectric 80.4\n$end\n\n$molecule')

    def test_from_file(self):
        from_file = QCTemplate.from_file(os.path.join(test_dir, "qin_good"))

        self.assertDictEqual(from_file.rem, {"job_type": "force",
                                             "basis": "def2-tzvppd",
                                             "max_scf_cycles": "200",
                                             "gen_scfman": "true",
                                             "xc_grid": "3",
                                             "scf_algorithm": "diis",
                                             "resp_charges": "true",
                                             "symmetry": "false",
                                             "sym_ignore": "true",
                                             "method": "wb97x-v",
                                             "solvent_method": "smd",
                                             "ideriv": "1",
                                             "thresh": "14"})
        self.assertDictEqual(from_file.smx, {"solvent": "other"})

    def test_from_qcinput(self):
        from_qcinp = QCTemplate.from_file(os.path.join(test_dir, "qchem_input.qin"))
        self.assertDictEqual(from_qcinp.rem, {'job_type': 'force',
                                              'basis': 'def2-tzvppd',
                                              'max_scf_cycles': '200',
                                              'gen_scfman': 'true',
                                              'scf_algorithm': 'diis',
                                              'method': 'wb97xd',
                                              'geom_opt_max_cycles': '200',
                                              'xc_grid': '3',
                                              'symmetry': 'false',
                                              'sym_ignore': 'true',
                                              'resp_charges': 'true'})


class TestGSMIsomerInput(PymatgenTest):

    def test_create(self):
        pass

    def test_verify_with_graphs(self):
        pass

    def test_str(self):
        pass

    def test_from_file(self):
        pass


if __name__ == "__main__":
    unittest.main()
