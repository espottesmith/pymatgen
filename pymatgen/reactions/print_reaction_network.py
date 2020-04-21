from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.util.testing import PymatgenTest
from pymatgen.reactions.reaction_network import (ReactionNetwork)
import os
from pymatgen.analysis.fragmenter import metal_edge_extender
import unittest
from monty.serialization import dumpfn, loadfn

test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        'test_files', 'reaction_network_files')


class TestReactionNetwork(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        EC_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "EC.xyz")),
            OpenBabelNN())
        cls.EC_mg = metal_edge_extender(EC_mg)

        LiEC_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "LiEC.xyz")),
            OpenBabelNN())
        cls.LiEC_mg = metal_edge_extender(LiEC_mg)

        LEDC_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "LEDC.xyz")),
            OpenBabelNN())
        cls.LEDC_mg = metal_edge_extender(LEDC_mg)

        LEMC_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "LEMC.xyz")),
            OpenBabelNN())
        cls.LEMC_mg = metal_edge_extender(LEMC_mg)

        H2O_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "H2O.xyz")),
            OpenBabelNN())
        cls.H2O_mg = metal_edge_extender(H2O_mg)

        cls.LiEC_extended_entries = []
        entries = loadfn(os.path.join(test_dir,"LiEC_extended_entries.json"))
        for entry in entries:
            mol = entry["output"]["optimized_molecule"]
            E = float(entry["output"]["final_energy"])
            H = float(entry["output"]["enthalpy"])
            S = float(entry["output"]["entropy"])
            mol_entry = MoleculeEntry(molecule=mol,energy=E,enthalpy=H,entropy=S,entry_id=entry["task_id"])
            cls.LiEC_extended_entries.append(mol_entry)

        cls.LiEC_reextended_entries = []
        entries = loadfn(os.path.join(test_dir,"LiEC_reextended_entries.json"))
        for entry in entries:
            if "optimized_molecule" in entry["output"]:
                mol = entry["output"]["optimized_molecule"]
            else:
                mol = entry["output"]["initial_molecule"]
            E = float(entry["output"]["final_energy"])
            H = float(entry["output"]["enthalpy"])
            S = float(entry["output"]["entropy"])
            mol_entry = MoleculeEntry(molecule=mol,energy=E,enthalpy=H,entropy=S,entry_id=entry["task_id"])
            if mol_entry.formula == "Li1":
                if mol_entry.charge == 1:
                    cls.LiEC_reextended_entries.append(mol_entry)
            else:
                cls.LiEC_reextended_entries.append(mol_entry)

        cls.RN = ReactionNetwork.from_input_entries(cls.LiEC_reextended_entries,
                                                electron_free_energy=-2.15)
        cls.RN.build()
        print(cls.RN.reactions)



if __name__ == "__main__":
    unittest.main()
