import unittest
import numpy as np
import matplotlib.pyplot as plt
import trtoolbox.svd as mysvd
from data_generator import DataGenerator


class TestSVD(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # generate data
        dgen = DataGenerator()
        dgen.gen_data()
        cls.data = dgen.data
        cls.wn = dgen.wn
        cls.time = dgen.time

        # prevents plt.show() from blocking execution
        plt.ion()

    def test_check_input(self):
        data, time, wn = mysvd.check_input(self.data, self.time, self.wn)
        self.assertEqual(data.shape, self.data.shape)

        data, time, wn = mysvd.check_input(self.data, self.time.T, self.wn)
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(wn.shape, self.wn.shape)
        self.assertEqual(time.shape[1], self.data.shape[1])

        data, time, wn = mysvd.check_input(self.data, self.time.T, self.wn.T)
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(time.shape[1], self.data.shape[1])
        self.assertEqual(wn.shape[0], self.data.shape[0])

        data, time, wn = mysvd.check_input(self.data.T, self.time, self.wn)
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(time.shape, self.time.shape)
        self.assertEqual(wn.shape, self.wn.shape)

        data, time, wn = mysvd.check_input(self.data.T, self.time.T, self.wn.T)
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(time.shape[1], self.data.shape[1])
        self.assertEqual(wn.shape[0], self.data.shape[0])

        data, time, wn = mysvd.check_input(
            self.data,
            self.time.flatten(),
            self.wn.flatten()
        )
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(time.shape[1], self.data.shape[1])
        self.assertEqual(wn.shape[0], self.data.shape[0])

        with self.assertRaises(ValueError):
            data, time, wn = mysvd.check_input(
                self.data,
                self.time,
                self.wn[2:]
            )

        with self.assertRaises(ValueError):
            data, time, wn = mysvd.check_input(
                self.data,
                self.time[2:],
                self.wn
            )

        with self.assertRaises(ValueError):
            data, time, wn = mysvd.check_input(
                self.data[2:, :-3],
                self.time,
                self.wn
            )

    def test_wrapper_svd(self):
        u, s, vt = mysvd.wrapper_svd(self.data)
        # checking dimensions
        self.assertEqual(u.shape[0], self.data.shape[0])
        self.assertEqual(s.shape[0], self.data.shape[0])
        self.assertEqual(vt.shape[0], self.data.shape[1])

        tdata = np.transpose(self.data)
        u, s, vt = mysvd.wrapper_svd(tdata)
        # checking dimensions
        self.assertEqual(u.shape[0], tdata.shape[0])
        self.assertEqual(s.shape[0], self.data.shape[0])
        self.assertEqual(vt.shape[0], tdata.shape[1])

    def test_show_svs(self):
        mysvd.show_svs(self.data, self.time, self.wn)
        plt.close('all')

    def test_reconstruct(self):
        res = mysvd.reconstruct(self.data, 5)
        # checking dimensions
        self.assertEqual(res.u.shape[0], self.data.shape[0])
        self.assertEqual(res.s.shape[0], self.data.shape[0])
        self.assertEqual(res.vt.shape[0], self.data.shape[1])
        self.assertEqual(res.svddata.shape, self.data.shape)

        tdata = np.transpose(self.data)
        res = mysvd.reconstruct(tdata, 5)
        # checking dimensions
        self.assertEqual(res.u.shape[0], tdata.shape[0])
        self.assertEqual(res.s.shape[0], self.data.shape[0])
        self.assertEqual(res.vt.shape[0], tdata.shape[1])
        self.assertEqual(res.svddata.shape, tdata.shape)

        n = [i+1 for i in range(5)]
        res = mysvd.reconstruct(self.data, n)

        n = [i for i in range(5)]
        with self.assertRaises(ValueError):
            res = mysvd.reconstruct(self.data, n)

        n = [1]*5
        with self.assertRaises(ValueError):
            res = mysvd.reconstruct(self.data, n)

    def test_dosvd(self):
        res = mysvd.dosvd(self.data, self.time, self.wn, 5)
        np.testing.assert_almost_equal(res.svddata, self.data)
