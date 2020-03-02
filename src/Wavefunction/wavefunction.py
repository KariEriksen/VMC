"""Wavefunction class."""
import math
import numpy as np


class Wavefunction:
    """Contains parameters of wavefunction and wave equation."""

    # deri_psi = 0.0
    # g        = 0.0
    # f        = 0.0

    def __init__(self, num_particles, num_dimensions, alpha, beta, a, system):
        """Instance of class."""
        self.num_p = num_particles
        self.num_d = num_dimensions
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.s = system

    def wavefunction(self, positions):
        """Return wave equation."""
        self.s.positions_distances(positions)
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

        pos = np.square(positions)
        if self.num_d > 2:
            pos[:, 2] *= self.beta
        pos_sum = np.sum(pos)
        g = np.exp(-self.alpha*pos_sum)
        # print (g)
        """
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
        # print (g)
        """
        return g

    def jastrow_factor(self, positions):
        """Calculate correlation factor."""
        f = 1.0

        for i in range(self.num_p):
            for j in range(i, self.num_p-1):

                distance = self.s.distances[i, j+1]
                # print ('d = ', distance)
                if distance > self.a:
                    f = f*(1.0 - (self.a/distance))
                else:
                    f *= 0.0
        return f

    def alpha_gradient_wavefunction(self, positions):
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

        n = self.num_p
        d = self.num_d
        a = self.a
        alpha = self.alpha
        beta = self.beta
        r_kj = np.zeros(d)
        d_psi_k = np.zeros(d)
        d_u_rkj = np.zeros(d)

        quantum_force = np.zeros((self.num_p, self.num_d))
        self.s.positions_distances(positions)

        for k in range(n):
            xk = positions[k, 0]
            yk = positions[k, 1]
            zk = positions[k, 2]
            rk = np.array((xk, yk, zk))

            d_psi_k[0] = -4*alpha*xk
            d_psi_k[1] = -4*alpha*yk
            d_psi_k[2] = -4*alpha*beta*zk

            for j in range(n):
                xj = positions[j, 0]
                yj = positions[j, 1]
                zj = positions[j, 2]
                rj = np.array((xj, yj, zj))

                if(j != k):
                    r_kj = rk - rj
                    # rkj = math.sqrt(np.sum((rk - rj)*(rk - rj)))
                    rkj = self.s.distances[k, j]
                    # factor = -2/(a*rkj*rkj - rkj*rkj*rkj)
                    factor = 2.0*a / math.pow(rkj, 1.5)
                    d_u_rkj += factor*r_kj

            quantum_force[k, :] = d_psi_k + d_u_rkj

        return quantum_force

    def quantum_force_numerical(self, positions):
        """Return drift force."""
        """This surely is inefficient, rewrite so the quantum force matrix
        gets updated, than calculating it over and over again each time"""
        quantum_force = np.zeros((self.num_p, self.num_d))
        position_forward = np.array(positions)
        psi_current = self.wavefunction(positions)
        psi_moved = 0.0
        step = 0.001

        for i in range(self.num_p):
            for j in range(self.num_d):
                position_forward[i, j] = position_forward[i, j] + step
                psi_moved = self.wavefunction(position_forward)
                # Resett positions
                position_forward[i, j] = position_forward[i, j] - step
                derivative = (psi_moved - psi_current)/step
                quantum_force[i, j] = (2.0/psi_current)*derivative

        return quantum_force
