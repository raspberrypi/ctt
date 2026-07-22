# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the CCM optimisation, in particular the flare compensation term.

import numpy as np

from ctt.algorithms.ccm import _DEFAULT_TEST_PATCHES, MACBETH_RGB, degamma, do_ccm, optimise_matrix
from ctt.utils.colorspace import rgb_to_lab


def _reference():
    """Reference patch colours in the i::6 order the fit uses (16-bit linear)."""
    m_srgb = degamma(np.array(MACBETH_RGB))
    m_lab = rgb_to_lab(m_srgb / 256)
    m_srgb = np.array([m_srgb[i::6] for i in range(6)]).reshape((24, 3))
    m_lab = np.array([m_lab[i::6] for i in range(6)]).reshape((24, 3))
    return m_srgb, m_lab


# A plausible rows-sum-to-1 ground-truth matrix.
M_TRUE = np.array(
    [
        [1.6, -0.4, -0.2],
        [-0.3, 1.5, -0.2],
        [-0.1, -0.5, 1.6],
    ]
)


def _synthetic_patches(flare_256=0.0):
    """Sensor patches that M_TRUE corrects exactly, plus uniform stray light.

    flare_256 is on the 0-256 scale used inside the optimiser; the returned
    channels are 16-bit like the real pipeline's.
    """
    m_srgb, m_lab = _reference()
    sensor = m_srgb @ np.linalg.inv(M_TRUE).T
    measured = sensor + flare_256 * 256.0
    return measured[:, 0], measured[:, 1], measured[:, 2], m_srgb, m_lab


class TestOptimiseMatrix:
    def test_recovers_matrix_without_flare(self):
        r, g, b, m_srgb, m_lab = _synthetic_patches()
        seed = do_ccm(r, g, b, m_srgb)
        mat, flare = optimise_matrix(seed, r, g, b, m_lab, 'average', list(_DEFAULT_TEST_PATCHES), False)
        assert flare == 0.0
        assert np.allclose(np.array(mat).reshape(3, 3), M_TRUE, atol=0.02)

    def test_flare_term_stays_near_zero_on_clean_data(self):
        r, g, b, m_srgb, m_lab = _synthetic_patches()
        seed = do_ccm(r, g, b, m_srgb)
        mat, flare = optimise_matrix(seed, r, g, b, m_lab, 'average', list(_DEFAULT_TEST_PATCHES), True)
        assert abs(flare) < 0.5
        assert np.allclose(np.array(mat).reshape(3, 3), M_TRUE, atol=0.02)

    def test_flare_compensation_recovers_flare_and_matrix(self):
        r, g, b, m_srgb, m_lab = _synthetic_patches(flare_256=3.0)
        seed = do_ccm(r, g, b, m_srgb)
        mat, flare = optimise_matrix(seed, r, g, b, m_lab, 'average', list(_DEFAULT_TEST_PATCHES), True)
        # The fitted offset should cancel the injected stray light...
        assert abs(flare + 3.0) < 0.5
        # ...leaving the matrix close to the ground truth.
        assert np.allclose(np.array(mat).reshape(3, 3), M_TRUE, atol=0.05)

    def test_flare_aware_matrix_beats_plain_fit_on_clean_scene(self):
        from colour.difference import delta_E_CIE2000

        r, g, b, m_srgb, m_lab = _synthetic_patches(flare_256=3.0)
        seed = do_ccm(r, g, b, m_srgb)
        patches = list(_DEFAULT_TEST_PATCHES)
        mat_plain, _ = optimise_matrix(seed, r, g, b, m_lab, 'average', patches, False)
        mat_flare, _ = optimise_matrix(seed, r, g, b, m_lab, 'average', patches, True)
        # Deployment condition: the scene has no calibration-rig flare.
        clean = (m_srgb @ np.linalg.inv(M_TRUE).T) / 256

        def mean_de(mat):
            lab = rgb_to_lab(clean @ np.array(mat).reshape(3, 3).T)
            return float(np.mean(delta_E_CIE2000(lab, m_lab)))

        assert mean_de(mat_flare) < mean_de(mat_plain)
