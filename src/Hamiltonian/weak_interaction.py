"""Hamiltonian class."""
import numpy as np
import math


class Weak_Interaction:
    """Calculate variables regarding the Hamiltonian of given wavefunction."""

    def __init__(self, omega, wavefunction, analytical):
        """Instance of class."""
        self.omega = omega
        self.s = wavefunction
        self.analytical = analytical

    def laplacian_numerical(self, positions):
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

    def laplacian_analytical(self, positions):
        """The analytical term for the laplacian, with weak interaction"""
        """In the weak interacting case the wavefunction is described by a
        harmonic oscillator with a trap potential"""
        """The analytic solution to kinetic energy given wave functions
        where a = scattering length"""

        d = self.s.num_d
        n = self.s.num_p
        a = self.s.a
        alpha = self.s.alpha
        alpha_sq = alpha**2
        beta = self.s.beta

        # rkj = np.zeros(d)
        # rki = np.zeros(d)
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

    def trap_potential_energy(self, positions):
        """Return the potential energy of the wavefunction."""
        """omega < 1.0 sylinder"""
        """omega = 1.0 symmetric"""
        """omega > 1.0 elliptic"""
        omega_sq = self.omega*self.omega

        # 0.5*omega_sq*np.sum(np.multiply(positions, positions))
        return 0.5*omega_sq*np.sum(np.multiply(positions, positions))

    def local_energy(self, positions):
        """Return the local energy."""
        if self.analytical == 'true':
            # Run with analytical expression for kinetic energy
            k = self.laplacian_analytical(positions)

        else:
            # Run with numerical expression for kinetic energy
            k = (self.laplacian_numerical(positions) /
                 self.s.wavefunction(positions))

        p = self.trap_potential_energy(positions)
        energy = -0.5*k + p

        return energy
