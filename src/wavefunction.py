"""Wavefunction class."""
import math
import numpy as np


class Wavefunction:
    """Contains parameters of wavefunction and wave equation."""

    # deri_psi = 0.0
    # g        = 0.0
    # f        = 0.0

    def __init__(self, num_particles, num_dimensions, alpha, beta, a):
        """Instance of class."""
        self.num_p = num_particles
        self.num_d = num_dimensions
        self.alpha = alpha
        self.beta = beta
        self.a = a

    def wavefunction(self, positions):
        """Return wave equation."""
        spf = self.single_particle_function(positions)
        jf = self.jastrow_factor(positions)
        wf = spf*jf

        return wf

    def single_particle_function(self, positions):
        """Return the single particle wave function."""
        """Take in position matrix of the particles and calculate the
        single particle wave function."""
        """Returns g, type float, product of all single particle wave functions
        of all particles."""

        g = 1.0

        for i in range(self.num_p):
            # self.num_d = j
            x = positions[i, 0]
            if self.num_d == 1:
                g = g*math.exp(-self.alpha*(x*x))
            elif self.num_d == 2:
                y = positions[i, 1]
                g = g*math.exp(-self.alpha*(x*x + y*y))
            else:
                y = positions[i, 1]
                z = positions[i, 2]
                g = g*math.exp(-self.alpha*(x*x + y*y + self.beta*z*z))
                # g = np.prod(math.exp(-self.alpha*(np.sum(
                # np.power(positions, 2)axis=1))))

        return g

    def jastrow_factor(self, positions):
        """Calculate correlation factor."""
        f = 1.0

        for i in range(self.num_p):
            for j in range(i, self.num_p-1):
                # ri_minus_rj = np.subtract(positions[i, :], positions[j+1, :])
                r = 0.0
                for k in range(self.num_d):
                    ri_minus_rj = positions[i, k] - positions[j+1, k]
                    r += ri_minus_rj**2
                distance = math.sqrt(r)
                # distance = math.sqrt(np.sum(np.square(ri_minus_rj)))
                if distance > self.a:
                    f = f*(1.0 - (self.a/distance))
                else:
                    f *= 1e-14
        return f

    def gradient_wavefunction(self, positions):
        """Calculate derivative of wave function divided by wave function."""
        """This expression holds for the case of the trail wave function
        described by the single particle wave function as a the harmonic
        oscillator function and the correlation function
        """
        deri_psi = 0.0

        for i in range(self.num_p):
            x = positions[i, 0]
            if self.num_d == 1:
                deri_psi += x*x
            elif self.num_d == 2:
                y = positions[i, 1]
                deri_psi += (x*x + y*y)
            else:
                y = positions[i, 1]
                z = positions[i, 2]
                deri_psi += (x*x + y*y + self.beta*z*z)

        return -deri_psi

    def wavefunction_ratio(self, positions, new_positions):
        """Wave function with new positions squared divided by."""
        """wave equation with old positions squared"""
        wf_old = self.wavefunction(positions)
        wf_new = self.wavefunction(new_positions)
        numerator = wf_new*wf_new
        denominator = wf_old*wf_old
        acceptance_ratio = numerator/denominator

        return acceptance_ratio

    def quantum_force(self, positions):
        """Return drift force."""
        """This surely is inefficient, rewrite so the quantum force matrix
        gets updated, than calculating it over and over again each time"""
        quantum_force = np.zeros((self.num_p, self.num_d))
        position_forward = np.array(positions)
        psi_current = self.wavefunction(positions)
        psi_moved = 0.0
        step = 0.001

        for i in range(self.s.num_p):
            for j in range(self.s.num_d):
                position_forward[i, j] = position_forward[i, j] + step
                psi_moved = self.wavefunction(position_forward)
                # Resett positions
                position_forward[i, j] = position_forward[i, j] - step
                derivative = (psi_moved - psi_current)/step
                quantum_force[i, j] = (2.0/psi_current)*derivative

        return quantum_force
