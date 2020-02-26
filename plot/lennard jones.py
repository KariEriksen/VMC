import numpy as np
import matplotlib.pyplot as plt # noqa: 401

r = np.arange(0.00001, 10, 0.005)

eps = 10.22
sigma = 2.556
# eps = 1.0
# sigma = 1.0
V = 4*eps*((sigma/r)**12 - (sigma/r)**6)


# plt.plot(e, n_BO, 'r--')
plt.plot(r, V, label='Lennard-Jones', color='mediumpurple')
plt.grid(color='gray', linestyle='-', linewidth=0.2)
# plt.hlines(0, 0, 10, color='dimgray')
plt.axis([2.3, 6, -12, 8])
# plt.axvline()
# locs, labels = xticks()
# plt.xticks(np.linspace(-10, 10, 5))
plt.xlabel('r')
plt.ylabel('V(r)')
# locs, labels = xticks()
plt.yticks([4, 0, -4, -8, -12])
plt.xticks([2.5, 3.5, 4.5, 5.5, 6.5])
# plt.title('Potential')
plt.legend(loc='upper right')
plt.show()
