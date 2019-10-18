"""Sampler class."""
import numpy as np
import math


class Sampler:
    """Calculate variables regarding energy of given wavefunction."""

    def __init__(self, omega, wavefunction):
        """Instance of class."""
        self.omega = omega
        self.s = wavefunction

    def laplacian(self, positions):
        """Numerical differentiation for solving laplacian."""

        step = 0.001
        position_forward = np.array(positions)
        position_backward = np.array(positions)
        psi_current = 0.0
        psi_moved = 0.0

        for i in range(self.s.num_p):
            psi_current += 2*self.s.num_d*self.s.wavefunction(positions)
            for j in range(self.s.num_d):

                position_forward[i, j] = position_forward[i, j] + step
                position_backward[i, j] = position_backward[i, j] - step
                wf_p = self.s.wavefunction(position_forward)
                wf_n = self.s.wavefunction(position_backward)
                psi_moved += wf_p + wf_n
                # Resett positions
                position_forward[i, j] = position_forward[i, j] - step
                position_backward[i, j] = position_backward[i, j] + step

        laplacian = (psi_moved - psi_current)/(step*step)
        return laplacian

    def laplacian_analytic(self, positions):
        """Assumes beta = 1.0"""

        c = 0.0
        d = self.s.num_d
        n = self.s.num_p
        for i in range(self.s.num_p):
            x = positions[i, 0]
            y = positions[i, 1]
            if d > 2:
                z = positions[i, 2]
                c += x**2 + y**2 + z**2
            else:
                c += x**2 + y**2

        laplacian_analytic = -2*d*n*self.s.alpha + 4*(self.s.alpha**2)*c

        return laplacian_analytic

    def laplacian_analytic_interaction(self, positions):
        """The analytical term for the laplacian, with interaction"""

        d = self.s.num_d
        n = self.s.num_p
        a = self.s.a
        alpha = self.s.alpha
        alpha_sq = alpha**2
        beta = self.s.beta

        rkj = np.zeros(d)
        rki = np.zeros(d)
        d_psi_rk = np.zeros(d)
        laplacian = 0.0

        for k in range(n):
            sum_2 = 0.0
            sum_3 = 0.0

            sum_1 = np.zeros(d)

            xk = positions[k, 0]
            yk = positions[k, 1]
            zk = positions[k, 2]
            rk = np.array((xk, yk, zk))

            d_psi_rk = [2*alpha*xk, 2*alpha*yk, 2*alpha*beta*zk]

            dd_psi_rk = (-4*alpha - 2*alpha*beta +
                         4*alpha_sq*(xk*xk + yk*yk + beta*zk*zk))

            for j in range(n):

                xj = positions[j, 0]
                yj = positions[j, 1]
                zj = positions[j, 2]
                rj = np.array((xj, yj, zj))

                if(j != k):

                    rkj = math.sqrt(np.sum((rk - rj)*(rk - rj)))

                    d_u_rkj = -a/(a*rkj - rkj*rkj)

                    xkj = xk - xj
                    ykj = yk - yj
                    zkj = zk - zj

                    sum_1[0] += ((xkj)/rkj)*d_u_rkj
                    sum_1[1] += ((ykj)/rkj)*d_u_rkj
                    sum_1[2] += ((zkj)/rkj)*d_u_rkj

                    dd_u_rkj = (a*(a - 2*rkj))/(rkj*rkj*(a - rkj)*(a - rkj))

                    sum_3 += dd_u_rkj + (2/rkj)*d_u_rkj

                    for i in range(n):

                        xi = positions[i, 0]
                        yi = positions[i, 1]
                        zi = positions[i, 2]
                        ri = np.array((xi, yi, zi))

                        if(i != k):

                            rki = math.sqrt(np.sum((rk - ri)*(rk - ri)))

                            d_u_rki = -a/(a*rki - rki*rki)

                            xki = xk - xi
                            yki = yk - yi
                            zki = zk - zi

                            rki_dot_rkj = xki*xkj + yki*ykj + zki*zkj

                            sum_2 += (rki_dot_rkj/(rki*rkj))*d_u_rki*d_u_rkj

            part = (d_psi_rk[0]*sum_1[0] + d_psi_rk[1]*sum_1[1]
                    + d_psi_rk[2]*sum_1[2])

            laplacian += dd_psi_rk + 2*part + sum_2 + sum_3

        return laplacian

    def potential_energy(self, positions):
        """Return the potential energy of the wavefunction."""
        omega_sq = self.omega*self.omega

        # 0.5*omega_sq*np.sum(np.multiply(positions, positions))
        return 0.5*omega_sq*np.sum(np.multiply(positions, positions))

    def local_energy(self, positions):
        """Return the local energy."""
        # Run with analytical expression for kinetic energy
        # k = -0.5*self.laplacian_analytic_interaction(positions)
        # Run with numerical expression for kinetic energy
        k = -0.5*self.laplacian(positions)/self.s.wavefunction(positions)
        p = self.potential_energy(positions)
        energy = k + p

        return energy

    def local_energy_times_wf(self, positions):
        """Return local energy times the derivative of wave equation."""
        energy = self.local_energy(positions)
        energy_times_wf = self.s.derivative_psi_term(positions)*energy

        return energy_times_wf

    def probability(self, positions, new_positions):
        """Wave function with new positions squared divided by."""
        """wave equation with old positions squared"""
        wf_old = self.s.wavefunction(positions)
        wf_new = self.s.wavefunction(new_positions)
        numerator = wf_new*wf_new
        denominator = wf_old*wf_old
        acceptance_ratio = numerator/denominator

        return acceptance_ratio

    def quantum_force(self, positions):
        """Return drift force."""
        """This surely is inefficient, rewrite so the quantum force matrix
        gets updated, than calculating it over and over again each time"""
        quantum_force = np.zeros((self.s.num_p, self.s.num_d))
        position_forward = np.array(positions)
        psi_current = self.s.wavefunction(positions)
        psi_moved = 0.0
        step = 0.001

        for i in range(self.s.num_p):
            for j in range(self.s.num_d):
                position_forward[i, j] = position_forward[i, j] + step
                psi_moved = self.s.wavefunction(position_forward)
                # Resett positions
                position_forward[i, j] = position_forward[i, j] - step
                derivative = (psi_moved - psi_current)/step
                quantum_force[i, j] = (2.0/psi_current)*derivative

        return quantum_force

    def greens_function(self, positions, new_positions, delta_t):
        """Calculate Greens function."""
        greens_function = 0.0

        D = 0.5
        F_old = self.quantum_force(positions)
        F_new = self.quantum_force(new_positions)
        for i in range(self.s.num_p):
            for j in range(self.s.num_d):
                term1 = 0.5*((F_old[i, j] + F_new[i, j]) *
                             (positions[i, j] - new_positions[i, j]))
                term2 = D*delta_t*(F_old[i, j] - F_new[i, j])
                greens_function += term1 + term2

        greens_function = np.exp(greens_function)

        return greens_function
