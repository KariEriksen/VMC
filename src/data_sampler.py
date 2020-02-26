"""Data sampler."""

import csv


class Data_Sampler:
    """Class for writing data to file"""

    def __init__(self, filename):
        """Instance of class."""
        self.filename = filename

    def make_file(self):
        """Create a new file"""
        # fix input stuff
        # f = open('/home/kari/VMC/data/data.csv', 'w')
        # f.write('alpha')

        if self.filename is None:
            with open('/home/kari/VMC/data/data.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["alpha", "energy", "d_energy"])

        else:
            with open('/home/kari/VMC/data/%s.csv' % self.filename,
                      'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["alpha", "energy", "d_energy"])

    def write_to_file(self):
        """Write average value to file"""

        # alpha from wave functions
        # energy, d_El from Sampler
        with open('/home/kari/VMC/data/data.csv', 'a', newline='') as file:
            alpha = self.w.alpha
            energy = self.s.local_energy
            d_El = self.s.derivative_energy
            writer = csv.writer(file)
            writer.writerow([alpha, energy, d_El])
