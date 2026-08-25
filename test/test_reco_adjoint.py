"""Shape and API checks for reco_adjoint (MRzeroCore-compatible)."""

from __future__ import annotations

import unittest

import numpy as np

from MRzeroCloud.reconstruction import reco_adjoint


class RecoAdjointTests(unittest.TestCase):
    def test_single_coil_shape(self):
        nx = ny = 8
        kx = np.fft.fftshift(np.fft.fftfreq(nx))
        ky = np.fft.fftshift(np.fft.fftfreq(ny))
        kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
        kspace = np.column_stack([kxg.ravel(), kyg.ravel(), np.zeros(nx * ny)])
        signal = np.ones(nx * ny, dtype=np.complex128)

        image = reco_adjoint(signal, kspace, resolution=(nx, ny, 1), FOV=(1.0, 1.0, 1.0))
        self.assertEqual(image.shape, (nx, ny, 1))
        self.assertTrue(np.iscomplexobj(image))

    def test_multicoil_rss_and_separate(self):
        n = 4
        kspace = np.zeros((n * n, 3))
        signal = np.ones((n * n, 2), dtype=np.complex128)
        rss = reco_adjoint(signal, kspace, resolution=(n, n, 1), FOV=(1.0, 1.0, 1.0))
        coils = reco_adjoint(
            signal,
            kspace,
            resolution=(n, n, 1),
            FOV=(1.0, 1.0, 1.0),
            return_multicoil=True,
        )
        self.assertEqual(rss.shape, (n, n, 1))
        self.assertEqual(coils.shape, (2, n, n, 1))


if __name__ == "__main__":
    unittest.main()
