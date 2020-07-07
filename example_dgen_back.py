from test.data_generator import DataGenerator
import matplotlib.pyplot as plt
import trtoolbox.lda as mylda
import trtoolbox.globalanalysis as mygf

dgen = DataGenerator()
dgen.gen_data(wnlimit=[1500, 1700], num_peaks=4, avg_width=40, noise=True, diff=True, back=True)
dgen.print_tcs()
dgen.plot_profile()
# dgen.plot_data()
# dgen.plot_das()

print('-----------')
res = mygf.doglobalfit(
    dgen.data, dgen.time, dgen.wn,
    [[4e-6, 3e-4, 2e-2], [4e-6, 9e-5, 2e-2]],
    method='est',
    back=True
)
res.plot_profile()
print('-----------')
res = mygf.doglobalfit(
    dgen.data, dgen.time, dgen.wn,
    [[4e-6, 3e-4, 2e-2], [4e-6, 9e-5, 2e-2]],
    method='raw',
    svds=3,
    back=True
)
res.plot_profile()
print('-----------')
res = mygf.doglobalfit(
    dgen.data, dgen.time, dgen.wn,
    [[4e-6, 3e-4, 2e-2], [4e-6, 9e-5, 2e-2]],
    method='svd',
    svds=3,
    back=True
)
res.plot_profile()

taus = [[4e-6, 3e-4, 2e-2], [4e-6, 9e-5, 2e-2]]
for i in range(3):
    print('-----------')
    res = mygf.doglobalfit(
        dgen.data, dgen.time, dgen.wn,
        taus,
        method='raw',
        svds=3,
        back=True
    )
    taus = res.tcs
    print('-----------')
    res = mygf.doglobalfit(
        dgen.data, dgen.time, dgen.wn,
        res.tcs,
        method='est',
        back=True
    )
    taus = res.tcs
res.plot_profile()

# res = mylda.dolda(dgen.data, dgen.time, dgen.wn, alimits=[0.1, 50])
# res.plot_results()

plt.show()
