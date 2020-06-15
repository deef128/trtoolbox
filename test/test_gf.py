import unittest
import numpy as np
import matplotlib.pyplot as plt
import trtoolbox.myglobalfit as mygf
import trtoolbox.mysvd as mysvd
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
        cls.tcs = dgen.tcs
        cls.profile = dgen.profile
        cls.das = dgen.das

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
            data, time, wn = mygf.check_input(
                self.data,
                self.time,
                self.wn[2:]
            )

        with self.assertRaises(ValueError):
            data, time, wn = mygf.check_input(
                self.data,
                self.time[2:],
                self.wn
            )

        with self.assertRaises(ValueError):
            data, time, wn = mygf.check_input(
                self.data[2:, :-3],
                self.time,
                self.wn
            )

    def test_model(self):
        s = [1, 0, 0]
        arr = mygf.model(s, self.time, 1/self.tcs)
        self.assertEqual(len(arr), len(s))

        arr = mygf.model(
            s,
            self.time,
            1/np.c_[self.tcs, self.tcs],
            back=True
        )
        self.assertEqual(len(arr), len(s))

    def test_create_profile(self):
        profile = mygf.create_profile(self.time, 1/self.tcs)
        np.testing.assert_array_equal(
            profile,
            self.profile
        )

        profile = mygf.create_profile(self.time.T, 1/self.tcs)
        np.testing.assert_array_equal(
            profile,
            self.profile
        )

        back_tcs = 1/np.c_[self.tcs, self.tcs]
        profile = mygf.create_profile(
            self.time,
            back_tcs,
            back=True
        )
        np.testing.assert_array_equal(
            profile.shape,
            self.profile.shape
        )

        profile = mygf.create_profile(
            self.time,
            back_tcs.T,
            back=True
        )
        np.testing.assert_array_equal(
            profile.shape,
            self.profile.shape
        )

        with self.assertRaises(ValueError):
            mygf.create_profile(
                self.time,
                np.ones((4, 5)),
                back=True
            )

        with self.assertRaises(ValueError):
            mygf.create_profile(
                self.time,
                1/self.tcs,
                back=True
            )

    def test_create_tr(self):
        pre = np.ones((self.tcs.size, 4))
        par = np.c_[pre, self.tcs]
        traces = mygf.create_tr(par, self.time)
        self.assertEqual(
            traces.shape,
            (pre.shape[1], self.time.size)
        )

    def test_create_das(self):
        das = mygf.create_das(self.profile, self.data)
        self.assertEqual(das.shape, self.das.shape)

    def test_calculate_fitdata(self):
        fitdata = mygf.calculate_fitdata(
            1/self.tcs,
            self.time,
            self.data
        )
        self.assertEqual(fitdata.shape, self.data.shape)

        fitdata = mygf.calculate_fitdata(
            1/self.tcs,
            self.time.T,
            self.data
        )
        self.assertEqual(fitdata.shape, self.data.shape)

        fitdata = mygf.calculate_fitdata(
            1/self.tcs,
            self.time,
            self.data.T
        )
        self.assertEqual(fitdata.shape, self.data.shape)

        fitdata = mygf.calculate_fitdata(
            1/np.c_[self.tcs, self.tcs],
            self.time,
            self.data,
            back=True
        )
        self.assertEqual(fitdata.shape, self.data.shape)

    def test_calculate_estimate(self):
        est = mygf.calculate_estimate(self.das, self.data)
        self.assertEqual(est.shape, self.profile.shape)

    def test_opt_func_raw(self):
        r = mygf.opt_func_raw(1/self.tcs, self.time, self.data)
        self.assertEqual(r.shape, self.data.flatten().shape)

        # back_tcs = 1/np.c_[self.tcs, self.tcs]
        # r = mygf.opt_func_raw(back_tcs, self.time, self.data, False)
        # self.assertEqual(r.shape, self.data.flatten().shape)

    def test_opt_func_est(self):
        r = mygf.opt_func_est(1/self.tcs, self.time, self.data)
        self.assertEqual(r.shape, self.profile.flatten().shape)

        # back_tcs = 1/np.c_[self.tcs, self.tcs]
        # r = mygf.opt_func_est(back_tcs, self.time, self.data)
        # self.assertEqual(r.shape, self.profile.flatten().shape)

    def test_opt_func_svd(self):
        pre = np.ones((self.tcs.size, 4))
        par = np.c_[pre, self.tcs]
        svds = 4
        u, s, vt = mysvd.wrapper_svd(self.data)
        sigma = np.zeros((u.shape[0], vt.shape[0]))
        sigma[:s.shape[0], :s.shape[0]] = np.diag(s)
        svdtraces = sigma[0:svds, :].dot(vt)
        r = mygf.opt_func_svd(
            par,
            self.time,
            self.data,
            svdtraces,
            self.tcs.size,
            False
        )
        self.assertEqual(r.shape, svdtraces.flatten().shape)

    def test_doglobalfit(self):
        print('\n Initial time constants')
        print(self.tcs)
        rdnm = 5 - 5 * np.random.random(self.tcs.shape)
        start_tcs = self.tcs * rdnm

        print('SVD')
        res = mygf.doglobalfit(
            self.data,
            self.time,
            self.wn,
            start_tcs,
            method='svd'
        )
        np.testing.assert_almost_equal(res.tcs, self.tcs, decimal=1)
        np.testing.assert_almost_equal(res.fitdata, self.data, decimal=5)

        print('Raw')
        res = mygf.doglobalfit(
            self.data,
            self.time,
            self.wn,
            start_tcs,
            method='raw'
        )
        np.testing.assert_almost_equal(res.tcs, self.tcs, decimal=0)
        np.testing.assert_almost_equal(res.fitdata, self.data, decimal=0)

        print('Est')
        res = mygf.doglobalfit(
            self.data,
            self.time,
            self.wn,
            start_tcs,
            method='est'
        )
        np.testing.assert_almost_equal(res.tcs, self.tcs, decimal=0)
        np.testing.assert_almost_equal(res.fitdata, self.data, decimal=0)

        dgen = DataGenerator()
        dgen.gen_data(back=True)
        print()
        dgen.print_tcs()
        rdnm = 5 - 5 * np.random.random(dgen.tcs.shape)
        start_tcs = dgen.tcs * rdnm

        print('SVD')
        res = mygf.doglobalfit(
            dgen.data,
            dgen.time,
            dgen.wn,
            start_tcs,
            method='svd',
            back=True
        )
        np.testing.assert_almost_equal(res.tcs, dgen.tcs, decimal=-1)
        np.testing.assert_almost_equal(res.fitdata, dgen.data, decimal=-1)

        with self.assertRaises(ValueError):
            res = mygf.doglobalfit(
                dgen.data,
                dgen.time,
                dgen.wn,
                start_tcs,
                method='raw',
                back=True
            )

        with self.assertRaises(ValueError):
            res = mygf.doglobalfit(
                dgen.data,
                dgen.time,
                dgen.wn,
                start_tcs,
                method='est',
                back=True
            )
