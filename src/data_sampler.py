"""Data sampler."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from sampler import Sampler # noqa: 401

class Data_Sampler:

    __init__(self, sample):

    self.sample = sample

    def make_file(self):
        """Create a new file"""
        # fix input stuff

        with open('../data/data.csv', 'w', newline='') as file:
            self.writer = csv.writer(file)
            self.writer.writerow(["alpha", "energy", "d_energy"])

    def write_to_file(self):
        """Write average value to file"""

        alpha = self.w.alpha
        energy = self.local_energy
        d_El = self.derivative_energy
        self.writer.writerow([parameter, energy, d_El])
