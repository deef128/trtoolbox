import trtoolbox.mylda as mylda
import numpy as np
import matplotlib.pyplot as plt

# load data
data = np.loadtxt('./data/data.dat', delimiter=',')
wn = np.loadtxt('./data/wavenumbers.dat', delimiter=',')
time = np.loadtxt('./data/time.dat', delimiter=',')

res = mylda.dolda(data, time, wn, tlimits=[1e-6, 1e-1])
# res.plotlda(alpha=0.9)
res.plot_results(alpha=0.9)
plt.show()
