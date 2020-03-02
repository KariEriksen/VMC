import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.optimize import curve_fit
from math import pi

sns.set_style(style='white')

data1 = pd.read_csv("../data/obd_int_data.csv")
data2 = pd.read_csv("../data/obd_non_data.csv")


def func(x, a, b, c):
    return a*np.exp(-(x-b)*(x-b)/(2*c*c))

x = data1['r']
y_int = data1['density']
y_non = data2['density']

x = np.array(x)

V = np.zeros(len(x))
V[0] = (4*pi/3)*x[1]**3
for i in range(1, len(x)-1):

    V[i] = (4*pi/3)*(x[i+1]**3 - x[i]**3)

V[40] = V[39]

y_int = y_int
y_non = y_non
# fit the interactive case
opt1, cov1 = curve_fit(func, x, y_int)
# fit the non-interactive case
opt2, cov2 = curve_fit(func, x, y_non)

# Plot
plt.plot(x, func(x, *opt1), 'blue', label = "a = 0.433")
plt.plot(x, func(x, *opt2), 'green', label = "a = 0.0")

plt.grid(color='gray', linestyle='-', linewidth=0.2)

plt.xlabel(r" $r$")
plt.ylabel(r"$\rho(r)$") 
# plt.xticks([0.1, 0.2, 0.3, 0.4])
# plt.axis([1.0, 4.0, 0.0, 0.2])

plt.legend()
plt.show()
