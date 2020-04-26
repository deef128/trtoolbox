import unittest
import numpy as np
import trtoolbox.mysvd as mysvd


class TestSVD(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = np.loadtxt('./data/data.dat', delimiter=',')

    def test_dosvd(self):
        u, s, vt = mysvd.dosvd(self.data)
        # checking dimensions
        self.assertEqual(u.shape[0], self.data.shape[0])
        self.assertEqual(s.shape[0], self.data.shape[0])
        self.assertEqual(vt.shape[0], self.data.shape[1])

    def test_reconstruct(self):
        res = mysvd.reconstruct(self.data, 5)
        # checking dimensions
        self.assertEqual(res.u.shape[0], self.data.shape[0])
        self.assertEqual(res.s.shape[0], self.data.shape[0])
        self.assertEqual(res.vt.shape[0], self.data.shape[1])
        self.assertEqual(res.svddata.shape, self.data.shape)
