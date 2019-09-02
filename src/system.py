"""System class."""
import math
import numpy as np


class System:
    """Contains parameters of system and wave equation."""

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
            y = positions[i, 1]
            if self.num_d > 2:
                # positions[i, 2] *= self.beta
                # if vector is 3 dimesions
                z = positions[i, 2]
                g = g*math.exp(-self.alpha*(x*x + y*y + self.beta*z*z))

            else:
                g = g*math.exp(-self.alpha*(x*x + y*y))
                # g = np.prod(math.exp(-self.alpha*(np.sum(
                # np.power(positions, 2)axis=1))))

        return g

    def jastrow_factor(self, positions):
        """Calculate correlation factor."""
        f = 1.0
        r = 0.0

        for i in range(self.num_p):
            for j in range(i, self.num_p-1):
                # ri_minus_rj = np.subtract(positions[i, :], positions[j+1, :])
                for k in range(self.num_d):
                    ri_minus_rj = positions[i, k] - positions[j+1, k]
                    r += ri_minus_rj**2
                distance = math.sqrt(r)
                # distance = math.sqrt(np.sum(np.square(ri_minus_rj)))
                if distance > self.a:
                    f *= 1.0 - (self.a/distance)
                else:
                    f *= 0
        return f

    def derivative_psi_term(self, positions):
        """Calculate derivative of wave function divided by wave function."""
        """This expression holds for the case of the trail wave function
        described by the single particle wave function as a the harmonic
        oscillator function and the correlation function
        """
        deri_psi = 0.0

        for i in range(self.num_p):
            x = positions[i, 0]
            y = positions[i, 1]
            if self.num_d > 2:
                # if vector is 3 dimesions
                # positions[i, 2] *= self.beta
                z = positions[i, 2]
                deri_psi += (x*x + y*y + self.beta*z*z)
            else:
                deri_psi += (x*x + y*y)

        return -deri_psi
