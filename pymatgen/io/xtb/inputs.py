# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.
import logging
import re
import ast
from typing import Dict
from monty.json import MSONable
from monty.io import zopen

# Classes for reading/manipulating/writing XTB Input and Output files

__author__ = "Evan Spotte-Smith, Shyam Dwaraknath"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"
__email__ = "ewcspottesmith@lbl.gov"

logger = logging.getLogger(__name__)


def xcontrol_eval(string):

    if "true" in string.lower():
        return True

    if "false" in string.lower():
        return False

    for val_type in (int, float):
        try:
            return val_type(string)
        except:
            pass

    return string


class XControl(MSONable):
    """
    An object representing a XTB Input file.
    """

    def __init__(self, blocks: Dict[str, Dict]):
        """
        Args:
            blocks: Dictionary mapping block names to blocks of input data
        """
        self.blocks = blocks

    def __repr__(self):
        data = []
        for block_name, block in self.blocks.items():
            data.append(f"${block_name.lower()}")
            for key, val in block.items():
                data.append(f"   {key.lower()} = {str(val)}")

        return "\n".join(data)

    @classmethod
    def from_file(cls, filename):
        """
        Builds an XControl object from a file
        """
        lines = []
        with zopen(filename) as f:
            lines = f.readlines()

        block_regex = re.compile(r"\s*\$(\w*)")
        data_regex = re.compile(r"\s*(\w.*)=\s*(.*)\s*\n")
        blocks = [
            (line_number, block_regex.match(line).group(1))
            for line_number, line in enumerate(lines)
            if "$" in line
        ]

        block_names = [b[1] for b in blocks]
        block_beginnings = [b[0] + 1 for b in blocks][:-1]
        block_endings = [b[0] for b in blocks][1:]

        block_data = {}
        for name, start, end in zip(block_names, block_beginnings, block_endings):
            block_data[name] = {}
            for line_number in range(start, end, 1):
                match = data_regex.match(lines[line_number])
                key = match.group(1)
                val = match.group(2)
                block_data[name][key] = xcontrol_eval(val.title())

        return cls(blocks=block_data)

