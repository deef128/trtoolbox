import trtoolbox.mysvd as mysvd
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# load data
data = np.loadtxt('./data/data.dat', delimiter=',')
wn = np.loadtxt('./data/wavenumbers.dat', delimiter=',')
wn = wn.reshape((wn.size, 1))
time = np.loadtxt('./data/time.dat', delimiter=',')
time = time.reshape((1, time.size))

# mysvd.show_svs(data, time, wn)
res = mysvd.reconstruct(data, [1, 2, 3, 5])
res.wn = wn
res.time = time
res.plot_results()
plt.show()

# for inspecting the Results class in Spyder after interactive plotting
# please run the clean() method!
# res.clean()
