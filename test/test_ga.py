# TODO: more testing on k-matrix
# TODO: errors on functions

import unittest
import numpy as np
import matplotlib.pyplot as plt
import trtoolbox.globalanalysis as mygf
import trtoolbox.svd as mysvd
from data_generator import DataGenerator


class TestGF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # generate data
        dgen = DataGenerator()
        dgen.gen_data()
        cls.data = dgen.data
        cls.wn = dgen.wn
        cls.time = dgen.time
        cls.tcs = dgen.rate_constants.tcs
        cls.ks = dgen.rate_constants.ks
        cls.rate_constants = dgen.rate_constants
        cls.profile = dgen.profile
        cls.das = dgen.sas

        dgen.gen_data(style='back')
        cls.tcs_back = dgen.rate_constants.tcs
        cls.rate_constants_back = dgen.rate_constants

        # prevents plt.show() from blocking execution
        plt.ion()

    def test_check_input(self):
        data, time, wn = mygf.check_input(self.data, self.time, self.wn)
        self.assertEqual(data.shape, self.data.shape)

        data, time, wn = mygf.check_input(self.data, self.time.T, self.wn)
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(wn.shape, self.wn.shape)
        self.assertEqual(time.shape[1], self.data.shape[1])

        data, time, wn = mygf.check_input(self.data, self.time.T, self.wn.T)
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(time.shape[1], self.data.shape[1])
        self.assertEqual(wn.shape[0], self.data.shape[0])

        data, time, wn = mygf.check_input(self.data.T, self.time, self.wn)
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(time.shape, self.time.shape)
        self.assertEqual(wn.shape, self.wn.shape)

        data, time, wn = mygf.check_input(self.data.T, self.time.T, self.wn.T)
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(time.shape[1], self.data.shape[1])
        self.assertEqual(wn.shape[0], self.data.shape[0])

        data, time, wn = mygf.check_input(
            self.data,
            self.time.flatten(),
            self.wn.flatten()
        )
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(time.shape[1], self.data.shape[1])
        self.assertEqual(wn.shape[0], self.data.shape[0])

        with self.assertRaises(ValueError):
            _ = mygf.check_input(
                self.data,
                self.time,
                self.wn[2:]
            )

        with self.assertRaises(ValueError):
            _ = mygf.check_input(
                self.data,
                self.time[2:],
                self.wn
            )

        with self.assertRaises(ValueError):
            _ = mygf.check_input(
                self.data[2:, :-3],
                self.time,
                self.wn
            )

    def test_model(self):
        s = [1, 0, 0]
        arr = mygf.model(s, self.time, self.rate_constants)
        self.assertEqual(len(arr), len(s))

        arr = mygf.model(
            s,
            self.time,
            self.rate_constants_back
        )
        self.assertEqual(len(arr), len(s))

    def test_create_profile(self):
        profile = mygf.create_profile(self.time, self.rate_constants)
        np.testing.assert_array_equal(
            profile,
            self.profile
        )

        profile = mygf.create_profile(self.time.T, self.rate_constants)
        np.testing.assert_array_equal(
            profile,
            self.profile
        )

        profile = mygf.create_profile(
            self.time,
            self.rate_constants_back
        )
        np.testing.assert_array_equal(
            profile.shape,
            self.profile.shape
        )

    def test_create_tr(self):
        pre = np.ones((self.tcs.size, 4))
        traces = mygf.create_tr(self.rate_constants, pre, self.time)
        self.assertEqual(
            traces.shape,
            (self.time.size, pre.shape[1])
        )

    def test_create_das(self):
        das = mygf.create_xas(self.profile, self.data)
        self.assertEqual(das.shape, self.das.shape)

    def test_calculate_fitdata(self):
        fitdata = mygf.calculate_fitdata(
            self.rate_constants,
            self.time,
            self.data
        )
        self.assertEqual(fitdata.shape, self.data.shape)

        fitdata = mygf.calculate_fitdata(
            self.rate_constants_back,
            self.time,
            self.data,
        )
        self.assertEqual(fitdata.shape, self.data.shape)

    def test_calculate_estimate(self):
        est = mygf.calculate_estimate(self.das, self.data)
        self.assertEqual(est.shape, self.profile.shape)

    def test_opt_func_raw(self):
        r = mygf.opt_func_raw(self.ks, self.rate_constants, self.time, self.data)
        self.assertEqual(r.shape, self.data.flatten().shape)

        # back_tcs = 1/np.c_[self.tcs, self.tcs]
        # r = mygf.opt_func_raw(back_tcs, self.time, self.data, False)
        # self.assertEqual(r.shape, self.data.flatten().shape)

    def test_opt_func_est(self):
        r = mygf.opt_func_est(self.ks, self.rate_constants, self.time, self.data)
        self.assertEqual(r.shape, self.profile.flatten().shape)

        # back_tcs = 1/np.c_[self.tcs, self.tcs]
        # r = mygf.opt_func_est(back_tcs, self.time, self.data)
        # self.assertEqual(r.shape, self.profile.flatten().shape)

    def test_opt_func_svd(self):
        pre = np.ones((self.tcs.size, 4))
        pars = np.hstack((self.rate_constants.ks, pre))
        svds = 4
        u, s, vt = mysvd.wrapper_svd(self.data)
        sigma = np.zeros((u.shape[0], vt.shape[0]))
        sigma[:s.shape[0], :s.shape[0]] = np.diag(s)
        svdtraces = sigma[0:svds, :].dot(vt)
        r = mygf.opt_func_svd(
            pars,
            self.rate_constants,
            self.time,
            svdtraces
        )
        self.assertEqual(r.shape, svdtraces.flatten().shape)

    def test_doglobalfit(self):
        print('\n Initial time constants')
        print(self.tcs)
        rdnm = 5 - 5 * np.random.random(self.tcs.shape)
        start_tcs = self.tcs * rdnm

        print('SVD')
        res = mygf.doglobalanalysis(
            self.data,
            self.time,
            self.wn,
            start_tcs,
            method='svd'
        )
        np.testing.assert_almost_equal(res.tcs, self.tcs, decimal=1)
        np.testing.assert_almost_equal(res.fitdata, self.data, decimal=4)

        print('Raw')
        res = mygf.doglobalanalysis(
            self.data,
            self.time,
            self.wn,
            start_tcs,
            method='raw'
        )
        np.testing.assert_almost_equal(res.tcs, self.tcs, decimal=0)
        np.testing.assert_almost_equal(res.fitdata, self.data, decimal=0)

        print('Est')
        res = mygf.doglobalanalysis(
            self.data,
            self.time,
            self.wn,
            start_tcs,
            method='est'
        )
        np.testing.assert_almost_equal(res.tcs, self.tcs, decimal=0)
        np.testing.assert_almost_equal(res.fitdata, self.data, decimal=0)

        dgen = DataGenerator()
        dgen.gen_data(style='back')
        print()
        dgen.print_tcs()
        rdnm = 5 - 5 * np.random.random(dgen.tcs.shape)
        start_tcs = dgen.tcs * rdnm

        print('SVD')
        res = mygf.doglobalanalysis(
            dgen.data,
            dgen.time,
            dgen.wn,
            start_tcs,
            method='svd',
            style='back'
        )
        # np.testing.assert_almost_equal(res.tcs, dgen.tcs, decimal=-1)
        np.testing.assert_almost_equal(res.fitdata, dgen.data, decimal=-1)

        # with self.assertRaises(ValueError):
        #     res = mygf.doglobalanalysis(
        #         dgen.data,
        #         dgen.time,
        #         dgen.wn,
        #         start_tcs,
        #         method='raw',
        #         back=True
        #     )
        #
        # with self.assertRaises(ValueError):
        #     res = mygf.doglobalanalysis(
        #         dgen.data,
        #         dgen.time,
        #         dgen.wn,
        #         start_tcs,
        #         method='est',
        #         back=True
        #     )
