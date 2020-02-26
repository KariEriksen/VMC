import numpy as np
import matplotlib.pyplot as plt

e_FD = np.arange(-10, 10, 0.0005)
e_BE = np.arange(0.1, 10, 0.0005)

kT = 1.0
n_FD = 1.0/(np.exp(e_FD) + 1.0)

n_BE = 1.0/(np.exp(e_BE/kT) - 1.0)

n_BO = 1.0/(np.exp(e_FD))


# plt.plot(e, n_BO, 'r--')
plt.plot(e_BE, n_BE, 'mediumpurple', e_FD, n_FD, 'yellowgreen',
         e_FD, n_BO, 'darkorange')
plt.axis([-10, 10, 0, 3])
# plt.axvline()
# locs, labels = xticks()
plt.xticks(np.linspace(-10, 10, 5))
plt.xlabel('($\epsilon-\mu$)/kT')
# plt.title('Distribution functions')
plt.text(-7, .7, 'Fermi-Dirac')
plt.text(-5, 2.3, 'Boltzmann')
plt.text(2, 1.7, 'Bose-Einstein')
plt.show()
