# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import trtoolbox.svd as mysvd
import trtoolbox.globalanalysis as mygf
import trtoolbox.lda as mylda
from trtoolbox.plothelper import PlotHelper

phelper = PlotHelper()

#%% laod data
data = np.loadtxt('data.dat', delimiter=',')
time = np.loadtxt('time.dat', delimiter=',')
wn = np.loadtxt('wavenumbers.dat', delimiter=',')

#%%
plt.close('all')
res = mygf.doglobalanalysis(data, time, wn, [1.4e-5, 1.2e-4, 2.1e-3, 2.3e-2], method='raw', svds=5)
res.plot_results()