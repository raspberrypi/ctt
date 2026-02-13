# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# CCM (colour correction matrix) calibration
#
# This program has many options from which to derive the color matrix from.
# The first is average. This minimises the average delta E across all patches of
# the macbeth chart. Testing across all cameras yielded this as the most color
# accurate and vivid. Other options are available however.
# Maximum minimises the maximum Delta E of the patches. It iterates through till
# a minimum maximum is found (so that there is not one patch that deviates wildly.)
# This yields generally good results but overall the colors are less accurate.
# The final option allows you to select the patches for which to average across.
# This means that you can bias certain patches, e.g. if you want the reds to be
# more accurate.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import colour.models
import numpy as np
from colour.difference import delta_E_CIE1976
from scipy.optimize import minimize

from ..detection.patches import get_alsc_patches
from ..output.visualise import visualise_macbeth_chart
from ..utils.colorspace import rgb_to_lab
from ..utils.tools import get_alsc_colour_cals, nudge_for_json
from .base import CalibrationAlgorithm

if TYPE_CHECKING:
    from ..core.camera import Camera

logger = logging.getLogger(__name__)


# Defaults when not supplied via config (e.g. tests). Config keys: ccm.matrix_selection, ccm.matrix_selection_types,
# ccm.test_patches.
_DEFAULT_MATRIX_SELECTION_TYPES = ('average', 'maximum', 'patches')
_DEFAULT_TEST_PATCHES = [1, 2, 5, 8, 9, 12, 14]


class CcmCalibration(CalibrationAlgorithm):
    json_key = 'rpi.ccm'

    def __init__(
        self,
        camera: Camera,
        platform: object,
        do_alsc_colour: bool = True,
        matrix_selection: str = 'average',
        test_patches: list[int] | None = None,
        matrix_selection_types: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(camera, platform)
        self.do_alsc_colour = do_alsc_colour
        allowed = matrix_selection_types if matrix_selection_types is not None else _DEFAULT_MATRIX_SELECTION_TYPES
        self.matrix_selection = matrix_selection if matrix_selection in allowed else 'average'
        self.test_patches = list(test_patches if test_patches is not None else _DEFAULT_TEST_PATCHES)

    def run(self) -> dict | None:
        cam = self.camera
        grid_size = self.platform.grid_size

        cam.log_new_sec('CCM')

        if cam.mono:
            logger.error("\nERROR: Can't do CCM on greyscale image!")
            cam.log += '\nERROR: Cannot perform CCM calibration '
            cam.log += 'on greyscale image!\nCCM aborted!'
            del cam.json['rpi.ccm']
            return None

        if ('rpi.alsc' not in cam.disable) and self.do_alsc_colour:
            colour_cals = get_alsc_colour_cals(cam.json)
            if colour_cals is not None:
                cam.log += '\nALSC tables found successfully'
            else:
                logger.warning('WARNING! No ALSC tables found for CCM!')
                logger.info('Performing CCM calibrations without ALSC correction...')
                cam.log += '\nWARNING: No ALSC tables found.\nCCM calibration '
                cam.log += 'performed without ALSC correction...'
        else:
            colour_cals = None
            cam.log += '\nWARNING: No ALSC tables found.\nCCM calibration '
            cam.log += 'performed without ALSC correction...'

        try:
            ccms = ccm(
                cam,
                colour_cals,
                grid_size,
                matrix_selection=self.matrix_selection,
                test_patches=self.test_patches,
            )
        except ArithmeticError:
            logger.error('ERROR: Matrix is singular!\nTake new pictures and try again...')
            cam.log += '\nERROR: Singular matrix encountered during fit!'
            cam.log += '\nCCM aborted!'
            return None

        cam.log += '\nCCM calibration written to json file'
        return {'ccms': ccms}


def degamma(x: np.ndarray) -> np.ndarray:
    """Takes 8-bit macbeth chart values, sRGB decode to linear and returns 16-bit."""
    x = np.asarray(x, dtype=float) / 255
    linear = colour.models.eotf_sRGB(x)
    return linear * 65535


def gamma(x: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma encoding. Accepts shape (3,) or (N, 3). Input 0–255, output 0–255."""
    x = np.asarray(x, dtype=float) / 255
    return colour.models.eotf_inverse_sRGB(x) * 255


def ccm(
    cam: Camera,
    colour_cals: dict | None,
    grid_size: tuple[int, int],
    *,
    matrix_selection: str = 'average',
    test_patches: list[int] | None = None,
) -> list[dict]:
    """Finds colour correction matrices for list of images."""
    imgs = cam.imgs
    # Standard macbeth chart colour values.
    m_rgb = np.array(
        [  # these are in RGB
            [116, 81, 67],  # dark skin
            [199, 147, 129],  # light skin
            [91, 122, 156],  # blue sky
            [90, 108, 64],  # foliage
            [130, 128, 176],  # blue flower
            [92, 190, 172],  # bluish green
            [224, 124, 47],  # orange
            [68, 91, 170],  # purplish blue
            [198, 82, 97],  # moderate red
            [94, 58, 106],  # purple
            [159, 189, 63],  # yellow green
            [230, 162, 39],  # orange yellow
            [35, 63, 147],  # blue
            [67, 149, 74],  # green
            [180, 49, 57],  # red
            [238, 198, 20],  # yellow
            [193, 84, 151],  # magenta
            [0, 136, 170],  # cyan (goes out of gamut)
            [245, 245, 243],  # white 9.5
            [200, 202, 202],  # neutral 8
            [161, 163, 163],  # neutral 6.5
            [121, 121, 122],  # neutral 5
            [82, 84, 86],  # neutral 3.5
            [49, 49, 51],  # black 2
        ]
    )
    # Convert reference colours from sRGB to linear RGB (now in 16-bit scale).
    m_srgb = degamma(m_rgb)
    # Produce array of LAB values for ideal color chart.
    m_lab = rgb_to_lab(m_srgb / 256)
    # Reorder reference values to match how patches are ordered.
    m_srgb = np.array([m_srgb[i::6] for i in range(6)]).reshape((24, 3))
    m_lab = np.array([m_lab[i::6] for i in range(6)]).reshape((24, 3))
    m_rgb = np.array([m_rgb[i::6] for i in range(6)]).reshape((24, 3))

    # For each image, perform AWB and ALSC corrections, then calculate the CCM for that image.
    ccm_tab = {}
    for img in imgs:
        cam.log += f'\nProcessing image: {img.name}'
        # Get macbeth patches with ALSC applied if ALSC enabled. If ALSC is disabled then
        # colour_cals will be None and the function will simply return the macbeth patches.
        r, b, g = get_alsc_patches(img, colour_cals, grey=False, grid_size=grid_size)
        # Do AWB. AWB is done by measuring the macbeth chart in the image, rather than from
        # the AWB calibration, so the AWB will be perfect and the CCM matrices more accurate.
        r_greys, b_greys, g_greys = r[3::4], b[3::4], g[3::4]
        r_g = np.mean(r_greys / g_greys)
        b_g = np.mean(b_greys / g_greys)
        r = r / r_g
        b = b / b_g
        # Normalise brightness wrt reference macbeth colours and then average each channel per patch.
        gain = np.mean(m_srgb) / np.mean((r, g, b))
        cam.log += f'\nGain with respect to standard colours: {gain:.3f}'
        r = np.mean(gain * r, axis=1)
        b = np.mean(gain * b, axis=1)
        g = np.mean(gain * g, axis=1)

        # Calculate CCM matrix.
        ccm_matrix = do_ccm(r, g, b, m_srgb)
        # Initial guess that the optimisation code works with.
        # CCM layout: [R1 R2 R3; G1 G2 G3; B1 B2 B3] * [R,G,B]' = out; optimising 6 elements, r3 = 1-r1-r2.
        original_ccm = ccm_matrix
        r1 = ccm_matrix[0]
        r2 = ccm_matrix[1]
        g1 = ccm_matrix[3]
        g2 = ccm_matrix[4]
        b1 = ccm_matrix[6]
        b2 = ccm_matrix[7]

        # Use the initial CCM as the guess for finding the optimised matrix.
        x0 = [r1, r2, g1, g2, b1, b2]
        _test_patches = test_patches if test_patches is not None else list(_DEFAULT_TEST_PATCHES)
        result = minimize(
            guess,
            x0,
            args=(r, g, b, m_lab, matrix_selection, _test_patches),
            tol=0.01,
        )
        # Produces a color matrix with the lowest delta E possible from the input data.
        # Note it is impossible for this to reach zero since the input data is imperfect.

        cam.log += '\n \n Optimised Matrix Below: \n \n'
        [r1, r2, g1, g2, b1, b2] = result.x
        # New optimised color correction matrix (rows sum to 1 to preserve greys).
        optimised_ccm = [r1, r2, (1 - r1 - r2), g1, g2, (1 - g1 - g2), b1, b2, (1 - b1 - b2)]

        cam.log += str(optimised_ccm)
        cam.log += '\n Old Color Correction Matrix Below \n'
        cam.log += str(ccm_matrix)

        formatted_ccm = np.array(original_ccm).reshape((3, 3))
        formatted_optimised_ccm = np.array(optimised_ccm).reshape((3, 3))

        # Apply CCM and get LAB for delta E; gamma version is for visualisation only.
        rgb_scaled = np.column_stack((r, g, b)) / 256
        optimised_ccm_rgb_arr = rgb_scaled @ formatted_ccm.T
        optimised_ccm_lab = rgb_to_lab(optimised_ccm_rgb_arr)
        after_optimised_rgb_arr = rgb_scaled @ formatted_optimised_ccm.T
        after_gamma_lab = rgb_to_lab(after_optimised_rgb_arr)

        # Clamp to valid range before gamma so visualisation doesn't get NaN
        optimised_ccm_rgb = np.clip(gamma(np.clip(optimised_ccm_rgb_arr, 0, 255)), 0, 255).astype(np.uint8).tolist()
        after_gamma_rgb = np.clip(gamma(np.clip(after_optimised_rgb_arr, 0, 255)), 0, 255).astype(np.uint8).tolist()

        cam.log += 'Here are the Improvements'

        before_metric = transform_and_evaluate(formatted_ccm, r, g, b, m_lab, matrix_selection, _test_patches)
        after_metric = transform_and_evaluate(formatted_optimised_ccm, r, g, b, m_lab, matrix_selection, _test_patches)
        old_worst_delta_e = float(np.max(deltae_array(optimised_ccm_lab, m_lab)))
        new_worst_delta_e = float(np.max(deltae_array(after_gamma_lab, m_lab)))
        metric_label = (
            'average delta E'
            if matrix_selection == 'average'
            else 'maximum delta E'
            if matrix_selection == 'maximum'
            else 'sum delta E (test patches)'
        )
        cam.log += (
            f'Before color correction matrix was optimised, {metric_label} = {before_metric}, '
            f'maximum delta E = {old_worst_delta_e}'
        )
        cam.log += (
            f'After color correction matrix was optimised, {metric_label} = {after_metric}, '
            f'maximum delta E = {new_worst_delta_e}'
        )

        # Top rectangle is ideal, left square is before optimisation, right square is after.
        visualise_macbeth_chart(
            m_rgb,
            optimised_ccm_rgb,
            after_gamma_rgb,
            str(cam.output_dir / f'{img.col}k'),
            matrix_selection=matrix_selection,
        )

        # If a CCM has already been calculated for this colour temperature, append; they are averaged later.
        if img.col in ccm_tab:
            ccm_tab[img.col].append(optimised_ccm)
        else:
            ccm_tab[img.col] = [optimised_ccm]
        cam.log += '\n'

    cam.log += '\nFinished processing images'
    # Average any CCMs that share a colour temperature.
    for k, v in ccm_tab.items():
        tab = np.mean(v, axis=0)
        ccm_tab[k] = list(nudge_for_json(tab, decimals=5))
        cam.log += f'\nMatrix calculated for colour temperature of {k} K'

    # Return all CCMs with respective colour temperature in the correct format, sorted by CT.
    sorted_ccms = sorted(ccm_tab.items(), key=lambda kv: kv[0])
    ccms = []
    for i in sorted_ccms:
        ccms.append({'ct': i[0], 'ccm': i[1]})
    return ccms


def guess(
    x0: list,
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    m_lab: np.ndarray,
    matrix_selection: str = 'average',
    test_patches: list[int] | None = None,
) -> float:
    """Provides numerical feedback for the optimisation: format CCM from 6 params and evaluate metric."""
    [r1, r2, g1, g2, b1, b2] = x0
    ccm_matrix = np.array([r1, r2, (1 - r1 - r2), g1, g2, (1 - g1 - g2), b1, b2, (1 - b1 - b2)]).reshape((3, 3))
    return transform_and_evaluate(ccm_matrix, r, g, b, m_lab, matrix_selection, test_patches)


def transform_and_evaluate(
    ccm_matrix: np.ndarray,
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    m_lab: np.ndarray,
    matrix_selection: str = 'average',
    test_patches: list[int] | None = None,
) -> float:
    """Transform colours to LAB via CCM and return the chosen metric (average/max/sum delta E)."""
    rgb_scaled = np.column_stack((r, g, b)) / 256
    rgb_post_ccm = rgb_scaled @ ccm_matrix.T  # RGB after color correction matrix
    lab_post_ccm = rgb_to_lab(rgb_post_ccm)
    de = deltae_array(lab_post_ccm, m_lab)
    if matrix_selection == 'average':
        return float(np.mean(de))
    if matrix_selection == 'maximum':
        return float(np.max(de))
    if matrix_selection == 'patches':
        patches = test_patches if test_patches is not None else _DEFAULT_TEST_PATCHES
        idx = np.asarray(patches, dtype=int)
        idx = np.clip(idx, 0, 23)
        return float(np.sum(de[idx]))
    return float(np.mean(de))


def deltae_array(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """Delta E (CIE 76) between corresponding rows; shapes (N, 3). Returns (N,) array."""
    return np.asarray(delta_E_CIE1976(lab_a, lab_b))


def sumde(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    """Sum of per-row Delta E between two (N, 3) Lab arrays."""
    return float(np.sum(deltae_array(lab_a, lab_b)))


def do_ccm(r: np.ndarray, g: np.ndarray, b: np.ndarray, m_srgb: np.ndarray) -> list:
    """
    Fit a 3x3 colour correction matrix (rows sum to 1) so CCM @ [r,g,b]' ≈ m_srgb
    in least-squares sense over the 24 patches. Uses (R-B, G-B) chrominance space
    so each row has two free coefficients; the third is 1 - first - second.
    Solves one 2x2 system per output channel: M @ [a, b]' = rhs, then c = 1 - a - b.
    """
    rb = r - b
    gb = g - b
    # 2x2 normal-equation matrix (same for all three output channels)
    M = np.array(
        [
            [np.sum(rb * rb), np.sum(rb * gb)],
            [np.sum(rb * gb), np.sum(gb * gb)],
        ]
    )
    # RHS per channel: 2x3 so that M @ X = RHS gives X (2x3) = [r_ab, g_ab, b_ab]
    RHS = np.array(
        [
            [np.sum(rb * (m_srgb[..., 0] - b)), np.sum(rb * (m_srgb[..., 1] - b)), np.sum(rb * (m_srgb[..., 2] - b))],
            [np.sum(gb * (m_srgb[..., 0] - b)), np.sum(gb * (m_srgb[..., 1] - b)), np.sum(gb * (m_srgb[..., 2] - b))],
        ]
    )
    det = np.linalg.det(M)
    # Raise error if matrix is singular; with real data this is rare—take new pictures if it happens.
    if abs(det) < 0.001:
        raise ArithmeticError
    # Solve M @ X = RHS  =>  X = M^{-1} @ RHS;  X is (2, 3), columns are (r_ab, g_ab, b_ab)
    X = np.linalg.solve(M, RHS)
    # Last row element per channel is 1 - first - second (rows sum to 1).
    ccm = np.column_stack([X[0], X[1], 1 - X[0] - X[1]]).T
    return ccm.ravel().tolist()


def deltae(color_a: list | np.ndarray, color_b: list | np.ndarray) -> float:
    """Delta E (CIE 76) between two Lab colours (length-3)."""
    return float(delta_E_CIE1976(np.asarray(color_a), np.asarray(color_b)))
