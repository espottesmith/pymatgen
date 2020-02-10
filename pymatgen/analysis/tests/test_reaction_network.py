# coding: utf-8


import os
import unittest
import time

from monty.serialization import dumpfn, loadfn

from pymatgen.core.structure import Molecule
from pymatgen.util.testing import PymatgenTest
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.entries.rxn_entry import ReactionEntry
from pymatgen.analysis.fragmenter import metal_edge_extender
from pymatgen.analysis.reaction_network import (ReactionNetwork,
                                                entries_from_reaction_label,
                                                generate_reaction_entries)

try:
    from openbabel import openbabel as ob
except ImportError:
    ob = None

__author__ = "Samuel Blau, Evan Spotte-Smith"
__email__ = "samblau1@gmail.com"

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

    def test_reextended(self):
        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries,
                                                electron_free_energy=-2.15)
        # self.assertEqual(len(RN.entries_list),569)
        # self.assertEqual(len(RN.graph.nodes),10481)
        # self.assertEqual(len(RN.graph.edges),22890)
        print(len(RN.entries_list))
        print(len(RN.graph.nodes))
        print(len(RN.graph.edges))

        EC_ind = None
        LEDC_ind = None
        LiEC_ind = None
        for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]
        self.assertEqual(EC_ind,456)
        self.assertEqual(LEDC_ind,511)
        self.assertEqual(Li1_ind,556)
        self.assertEqual(LiEC_ind,424)
        # print(EC_ind)
        # print(LEDC_ind)
        # print(Li1_ind)
        # print(LiEC_ind)

        PR_paths, paths = RN.find_paths([EC_ind,Li1_ind],LEDC_ind,weight="softplus",num_paths=10)
        # PR_paths, paths = RN.find_paths([LiEC_ind],LEDC_ind,weight="softplus",num_paths=10)
        # PR_paths, paths = RN.find_paths([LiEC_ind],42,weight="softplus",num_paths=10)
        # PR_paths, paths = RN.find_paths([EC_ind,Li1_ind],LiEC_ind,weight="softplus",num_paths=10)
        # PR_paths, paths = RN.find_paths([EC_ind,Li1_ind],42,weight="exponent",num_paths=10)
        for path in paths:
            for val in path:
                print(val, path[val])
            print()

    def _test_build_graph(self):
        RN = ReactionNetwork.from_input_entries(self.LiEC_extended_entries,
                                                electron_free_energy=-2.15)
        self.assertEqual(len(RN.entries_list),251)
        self.assertEqual(len(RN.graph.nodes),2021)
        self.assertEqual(len(RN.graph.edges),4022)
        # dumpfn(RN,"RN.json")
        loaded_RN = loadfn("RN.json")
        self.assertEqual(RN.as_dict(),loaded_RN.as_dict())

    def _test_solve_prerequisites(self):
        RN = loadfn("RN.json")
        LiEC_ind = None
        LEDC_ind = None
        for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        PRs = RN.solve_prerequisites([LiEC_ind],LEDC_ind,weight="softplus")
        # dumpfn(PRs,"PRs.json")
        loaded_PRs = loadfn("PRs.json")
        for key in PRs:
            self.assertEqual(PRs[key],loaded_PRs[str(key)])

    # def _test_solve_multi_prerequisites(self):
    #     RN = loadfn("RN.json")
    #     LiEC_ind = None
    #     LEDC_ind = None
    #     for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
    #         if self.LiEC_mg.isomorphic_to(entry.mol_graph):
    #             LiEC_ind = entry.parameters["ind"]
    #             break
    #     for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
    #         if self.LEDC_mg.isomorphic_to(entry.mol_graph):
    #             LEDC_ind = entry.parameters["ind"]
    #             break
    #     # print(RN.entries["C1 O1"][1][0][0])
    #     CO_ind = RN.entries["C1 O1"][1][0][0].parameters["ind"]
    #     print(CO_ind)
    #     PRs = RN.solve_prerequisites([LiEC_ind,CO_ind],LEDC_ind,weight="softplus")
    #     # dumpfn(PRs,"PRs.json")
    #     # loaded_PRs = loadfn("PRs.json")
    #     # for key in PRs:
    #     #     self.assertEqual(PRs[key],loaded_PRs[str(key)])

    def _test_find_paths(self):
        RN = loadfn("RN.json")
        LiEC_ind = None
        LEDC_ind = None
        for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        PR_paths, paths = RN.find_paths([LiEC_ind],LEDC_ind,weight="softplus",num_paths=10)
        self.assertEqual(paths[0]["cost"],1.7660275897855464)
        self.assertEqual(paths[0]["overall_free_energy_change"],-5.131657887139409)
        self.assertEqual(paths[0]["hardest_step_deltaG"],0.36044270861384575)
        self.assertEqual(paths[9]["cost"],3.7546340395839226)
        self.assertEqual(paths[9]["overall_free_energy_change"],-5.13165788713941)
        self.assertEqual(paths[9]["hardest_step_deltaG"],2.7270388301945787)

    def test_as_from_dict(self):
        orig = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries,
                                                  electron_free_energy=-2.15)

        rn_dict = orig.as_dict()

        rn_from_dict = ReactionNetwork.from_dict(rn_dict)

        self.assertEqual(len(rn_from_dict.entries_list), len(orig.entries_list))

    # def _test_find_multi_paths(self):
    #     RN = loadfn("RN.json")
    #     LiEC_ind = None
    #     LEDC_ind = None
    #     for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
    #         if self.LiEC_mg.isomorphic_to(entry.mol_graph):
    #             LiEC_ind = entry.parameters["ind"]
    #             break
    #     for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
    #         if self.LEDC_mg.isomorphic_to(entry.mol_graph):
    #             LEDC_ind = entry.parameters["ind"]
    #             break
    #     CO_ind = 29
    #     PR_paths, paths = RN.find_paths([LiEC_ind,CO_ind],LEDC_ind,weight="softplus",num_paths=10)
    #     self.assertEqual(paths[0]["cost"],1.7660275897855464)
    #     self.assertEqual(paths[0]["overall_free_energy_change"],-5.131657887139409)
    #     self.assertEqual(paths[0]["hardest_step_deltaG"],0.36044270861384575)
    #     self.assertEqual(paths[9]["cost"],3.7546340395839226)
    #     self.assertEqual(paths[9]["overall_free_energy_change"],-5.13165788713941)
    #     self.assertEqual(paths[9]["hardest_step_deltaG"],2.7270388301945787)


class TestReactionNetworkUtils(PymatgenTest):

    def setUp(self) -> None:
        self.maxDiff = None

        comp_entries = loadfn(os.path.join(test_dir, "single_completed_reaction.json"))
        self.reference = ReactionEntry(comp_entries["rcts"],
                                       comp_entries["pros"],
                                       transition_state=comp_entries["ts"])

        entries = loadfn(os.path.join(test_dir, "LiEC_extended_entries.json"))
        extended_entries = list()
        for entry in entries:
            if "optimized_molecule" in entry["output"]:
                mol = entry["output"]["optimized_molecule"]
            else:
                mol = entry["output"]["initial_molecule"]
            e = float(entry["output"]["final_energy"])
            h = float(entry["output"]["enthalpy"])
            s = float(entry["output"]["entropy"])
            mol_entry = MoleculeEntry(molecule=mol, energy=e, enthalpy=h, entropy=s,
                                      entry_id=entry["task_id"])
            extended_entries.append(mol_entry)

        self.network = ReactionNetwork.from_input_entries(extended_entries,
                                                          electron_free_energy=-2.15)
        self.entries = extended_entries
        self.rxn_nodes = {n for n, d in self.network.graph.nodes(data=True) if d['bipartite'] == 1}

        # self.generate_test_files()

    def generate_test_files(self):

        rxn_to_mol = dict()
        for rxn_node in self.rxn_nodes:
            rct_entries, pro_entries = entries_from_reaction_label(self.network,
                                                                   rxn_node)
            rxn_to_mol[rxn_node] = {"reactants": rct_entries,
                                    "products": pro_entries}

        dumpfn(rxn_to_mol, os.path.join(test_dir,
                                        "rxn_labels_to_mol_entries.json"))

        all_rxn_entries = generate_reaction_entries(self.network,
                                                    universal_reference=self.reference)
        dumpfn(all_rxn_entries, os.path.join(test_dir,
                                             "all_rxn_entries.json"))

    def test_entries_from_reaction_label(self):
        rxn_to_mol = loadfn(os.path.join(test_dir,
                                         "rxn_labels_to_mol_entries.json"))

        for rxn_node in self.rxn_nodes:
            rct_entries, pro_entries = entries_from_reaction_label(self.network,
                                                                   rxn_node)
            rct_ids = set([e.entry_id for e in rct_entries])
            pro_ids = set([e.entry_id for e in pro_entries])

            self.assertSetEqual(rct_ids,
                                set([e.entry_id for e in rxn_to_mol[rxn_node]["reactants"]]))
            self.assertSetEqual(pro_ids,
                                set([e.entry_id for e in rxn_to_mol[rxn_node]["products"]]))

    def test_generate_reaction_entries(self):

        all_rxn_entries = loadfn(os.path.join(test_dir, "all_rxn_entries.json"))

        for rxn_node in self.rxn_nodes:
            base_label = rxn_node.replace("PR_", "")

            self.assertTrue(base_label in all_rxn_entries)

            mol_entries = entries_from_reaction_label(self.network, rxn_node)
            entry = ReactionEntry(mol_entries[0], mol_entries[1],
                                  reference_reaction=self.reference,
                                  approximate_method="EBEP",
                                  entry_id=rxn_node)
            self.assertEqual(str(entry), str(all_rxn_entries[base_label]))


if __name__ == "__main__":
    unittest.main()
