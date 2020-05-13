import trtoolbox.mysvd as mysvd
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# load data
data = np.loadtxt('./data/data.dat', delimiter=',')
wn = np.loadtxt('./data/wavenumbers.dat', delimiter=',')
time = np.loadtxt('./data/time.dat', delimiter=',')

# if time and frequency are manually added
# time has to span columns and frequency rows.
# wn = wn.reshape((wn.size, 1))
# time = time.reshape((1, time.size))

res = mysvd.dosvd(data, time, wn)
plt.show()

# for inspecting the Results class in Spyder after interactive plotting
# please run the clean() method!
# res.clean()
