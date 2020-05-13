import trtoolbox.myglobalfit as mygf
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# load data
data = np.loadtxt('./data/data.dat', delimiter=',')
wn = np.loadtxt('./data/wavenumbers.dat', delimiter=',')
time = np.loadtxt('./data/time.dat', delimiter=',')

tcs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
res = mygf.doglobalfit(data, time, wn, tcs, svds=5)
res.plot_results()
# res.plot_traces()
# res.plot_spectra()
plt.show()

# for inspecting the Results class in Spyder after interactive plotting
# please run the clean() method!
# res.clean()
