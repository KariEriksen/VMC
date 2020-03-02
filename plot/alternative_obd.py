import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import norm

sns.set_style(style='white')

data1 = pd.read_csv("../data/obd_int_data.csv")
data2 = pd.read_csv("../data/obd_non_data.csv")

x = data1['r']
y_int = data1['density']*10
y_non = data2['density']*10

mu1, std1 = norm.fit(y_int)
mu2, std2 = norm.fit(y_non)

p1 = norm.pdf(x, mu1, std1)
p2 = norm.pdf(x, mu2, std2)

plt.grid(color='gray', linestyle='-', linewidth=0.2)
plt.plot(x, p1, 'blue', label = "a = 0.433")
plt.plot(x, p2, 'green', label = "a = 0.0")

plt.xlabel(r" $r$")
plt.ylabel(r"$\rho(r)$") 
# plt.xticks([0.1, 0.2, 0.3, 0.4])
plt.axis([0.0, 3.0, 0.0, 1.0])

plt.legend()
plt.show()
