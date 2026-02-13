# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Macbeth chart locator

from __future__ import annotations

import logging
import warnings
from importlib import resources
from typing import TYPE_CHECKING

import cv2
import numpy as np
from sklearn import cluster as cluster

from ..utils.errors import MacbethError
from ..utils.tools import correlate, reshape
from .ransac import get_square_centres, get_square_verts

if TYPE_CHECKING:
    from ..core.camera import Camera

logger = logging.getLogger(__name__)


def fxn():
    warnings.warn('runtime', RuntimeWarning, stacklevel=2)


success_msg = 'Macbeth chart located successfully'


def find_macbeth(cam: Camera, img: np.ndarray, mac_config: tuple[int, int] = (0, 0)) -> tuple | None:
    small_chart, show = mac_config
    cam.log += '\nLocating macbeth chart'
    warnings.simplefilter('ignore')
    fxn()

    ref_path = resources.files('ctt.data') / 'ctt_ref.pgm'
    with resources.as_file(ref_path) as p:
        ref = cv2.imread(str(p), flags=cv2.IMREAD_GRAYSCALE)
    ref_w = 120
    ref_h = 80
    rc1 = (0, 0)
    rc2 = (0, ref_h)
    rc3 = (ref_w, ref_h)
    rc4 = (ref_w, 0)
    ref_corns = np.array((rc1, rc2, rc3, rc4), np.float32)
    ref_data = (ref, ref_w, ref_h, ref_corns)

    cor, mac, coords, msg = get_macbeth_chart(img, ref_data)

    all_images = [img]

    if cor < 0.75:
        a = 2
        img_br = cv2.convertScaleAbs(img, alpha=a, beta=0)
        all_images.append(img_br)
        cor_b, mac_b, coords_b, msg_b = get_macbeth_chart(img_br, ref_data)
        if cor_b > cor:
            cor, _mac, coords, msg = cor_b, mac_b, coords_b, msg_b

    if cor < 0.75:
        a = 4
        img_br = cv2.convertScaleAbs(img, alpha=a, beta=0)
        all_images.append(img_br)
        cor_b, mac_b, coords_b, msg_b = get_macbeth_chart(img_br, ref_data)
        if cor_b > cor:
            cor, _mac, coords, msg = cor_b, mac_b, coords_b, msg_b

    ii = -1
    w_best = 0
    h_best = 0
    d_best = 100
    if cor != 0:
        d_best = 0

    # Grid search at multiple scales to locate chart. Each successive search uses smaller
    # windows (w_frac, h_frac) and finer grid steps. Blocks 3-4 are conditional and only
    # run for small charts when prior searches didn't succeed.
    # Config: (w_frac, h_frac, steps, inc_divisor, d_value, images)
    search_configs = [
        (2 / 3, 2 / 3, 3, 6, 1, all_images),  # Coarse 3×3 grid
        (1 / 2, 1 / 2, 5, 8, 2, all_images),  # Medium 5×5 grid
        (1 / 3, 1 / 3, 9, 12, 3, [img]),  # Fine 9×9 grid (small charts only)
        (1 / 4, 1 / 4, 13, 16, None, [img]),  # Finest 13×13 grid (small charts only)
    ]

    for idx, (w_frac, h_frac, steps, inc_divisor, d_value, images) in enumerate(search_configs):
        # Blocks 3-4 are conditional on small_chart and prior d_best threshold
        if idx == 2 and not (small_chart and cor < 0.75 and d_best > 1):
            continue
        if idx == 3 and not (small_chart and cor < 0.75 and d_best > 2):
            continue
        if cor >= 0.75:
            break

        shape = list(img.shape[:2])
        w, h = shape
        w_sel = int(w_frac * w)
        h_sel = int(h_frac * h)
        w_inc = int(w / inc_divisor)
        h_inc = int(h / inc_divisor)

        for img_br in images:
            for i in range(steps):
                for j in range(steps):
                    w_s, h_s = i * w_inc, j * h_inc
                    img_sel = img_br[w_s : w_s + w_sel, h_s : h_s + h_sel]
                    cor_ij, mac_ij, coords_ij, msg_ij = get_macbeth_chart(img_sel, ref_data)
                    if cor_ij > cor:
                        cor = cor_ij
                        _mac, coords, msg = mac_ij, coords_ij, msg_ij
                        ii, jj = i, j
                        w_best, h_best = w_inc, h_inc
                        if d_value is not None:
                            d_best = d_value

    if ii != -1:
        for a in range(len(coords)):
            for b in range(len(coords[a][0])):
                coords[a][0][b][1] += ii * w_best
                coords[a][0][b][0] += jj * h_best

    coords_fit = None
    cam.log += f'\n{msg}'
    if msg == success_msg:
        coords_fit = coords
        cam.log += '\nMacbeth chart vertices:\n'
        cam.log += f'{2 * np.round(coords_fit[0][0])}'
        cam.log += f'\nConfidence: {cor:.3f}'
        if cor < 0.75:
            logger.info('Caution: Low confidence guess!')
            cam.log += 'WARNING: Low confidence guess!'
    else:
        logger.info(msg)

    if show and coords_fit is not None:
        copy = img.copy()
        verts = coords_fit[0][0]
        cents = coords_fit[1][0]

        for vert in verts:
            p = tuple(np.round(vert).astype(np.int32))
            cv2.circle(copy, p, 10, 1, -1)
        for i in range(len(cents)):
            cent = cents[i]
            p = tuple(np.round(cent).astype(np.int32))
            if i == 3:
                cv2.circle(copy, p, 8, 0, -1)
            elif i == 23:
                cv2.circle(copy, p, 8, 1, -1)
            else:
                cv2.circle(copy, p, 8, 0.5, -1)
        copy, _ = reshape(copy, 400)

    return (coords_fit, cor if msg == success_msg else None)


def get_macbeth_chart(img: np.ndarray, ref_data: tuple) -> tuple:
    (ref, ref_w, ref_h, ref_corns) = ref_data

    try:
        src = img
        src, factor = reshape(src, 200)
        original = src.copy()
        a = 125 / np.average(src)
        src_norm = cv2.convertScaleAbs(src, alpha=a, beta=0)
        src_bw = cv2.cvtColor(src_norm, cv2.COLOR_BGR2GRAY) if len(src_norm.shape) == 3 else src_norm
        original_bw = src_bw.copy()
        sigma = 2
        src_bw = cv2.GaussianBlur(src_bw, (0, 0), sigma)
        t1, t2 = 50, 100
        edges = cv2.Canny(src_bw, t1, t2)
        k_size = 2
        kernel = np.ones((k_size, k_size))
        its = 1
        edges = cv2.dilate(edges, kernel, iterations=its)
        conts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(conts) == 0:
            raise MacbethError(
                '\nWARNING: No macbeth chart found!'
                '\nNo contours found in image\n'
                'Possible problems:\n'
                '- Macbeth chart is too dark or bright\n'
                '- Macbeth chart is occluded\n'
            )
        epsilon = 0.07
        conts_per = []
        for i in range(len(conts)):
            per = cv2.arcLength(conts[i], True)
            poly = cv2.approxPolyDP(conts[i], epsilon * per, True)
            if len(poly) == 4 and cv2.isContourConvex(poly):
                conts_per.append((poly, per))

        if len(conts_per) == 0:
            raise MacbethError(
                '\nWARNING: No macbeth chart found!'
                '\nNo quadrilateral contours found'
                '\nPossible problems:\n'
                '- Macbeth chart is too dark or bright\n'
                '- Macbeth chart is occluded\n'
                '- Macbeth chart is out of camera plane\n'
            )

        conts_per = sorted(conts_per, key=lambda x: x[1])
        med_per = conts_per[int(len(conts_per) / 2)][1]
        side = med_per / 4
        perc = 0.1
        med_low, med_high = med_per * (1 - perc), med_per * (1 + perc)
        squares = []
        for i in conts_per:
            if med_low <= i[1] and med_high >= i[1]:
                squares.append(i[0])

        square_verts, mac_norm = get_square_verts(0.06)
        mac_mids = []
        squares_raw = []
        for i in range(len(squares)):
            square = squares[i]
            squares_raw.append(square)
            rect = cv2.minAreaRect(square)
            square = cv2.boxPoints(rect).astype(np.float32)
            square = sorted(square, key=lambda x: x[0])
            square_1 = sorted(square[:2], key=lambda x: x[1])
            square_2 = sorted(square[2:], key=lambda x: -x[1])
            square = np.array(np.concatenate((square_1, square_2)), np.float32)
            square = np.reshape(square, (4, 2)).astype(np.float32)
            squares[i] = square
            for j in range(len(square_verts)):
                verts = square_verts[j]
                p_mat = cv2.getPerspectiveTransform(verts, square)
                mac_guess = cv2.perspectiveTransform(mac_norm, p_mat)
                mac_guess = np.round(mac_guess).astype(np.int32)
                in_border = True

                if in_border:
                    mac_mid = np.mean(mac_guess, axis=1)
                    mac_mids.append([mac_mid, (i, j)])

        if len(mac_mids) == 0:
            raise MacbethError(
                '\nWARNING: No macbeth chart found!'
                '\nNo possible macbeth charts found within image'
                '\nPossible problems:\n'
                '- Part of the macbeth chart is outside the image\n'
                '- Quadrilaterals in image background\n'
            )

        for i in range(len(mac_mids)):
            mac_mids[i][0] = mac_mids[i][0][0]

        clustering = cluster.AgglomerativeClustering(
            n_clusters=None, compute_full_tree=True, distance_threshold=side * 2
        )
        mac_mids_list = [x[0] for x in mac_mids]

        if len(mac_mids_list) == 1:
            clus_list = []
            clus_list.append([mac_mids, len(mac_mids)])

        else:
            clustering.fit(mac_mids_list)

            clus_list = []
            if clustering.n_clusters_ > 1:
                for i in range(clustering.labels_.max() + 1):
                    indices = [j for j, x in enumerate(clustering.labels_) if x == i]
                    clus = []
                    for index in indices:
                        clus.append(mac_mids[index])
                    clus_list.append([clus, len(clus)])
                clus_list.sort(key=lambda x: -x[1])

            elif clustering.n_clusters_ == 1:
                clus_list.append([mac_mids, len(mac_mids)])
            else:
                raise MacbethError('\nWARNING: No macebth chart found!\nNo clusters found\nPossible problems:\n- NA\n')

        clus_len_max = clus_list[0][1]
        clus_tol = 0.7
        for i in range(len(clus_list)):
            if clus_list[i][1] < clus_len_max * clus_tol:
                clus_list = clus_list[:i]
                break
            cent = np.mean(clus_list[i][0], axis=0)[0]
            clus_list[i].append(cent)

        reference = get_square_centres(0.06)

        max_cor = 0
        best_fit = None
        best_cen_fit = None
        best_ref_mat = None

        for clus in clus_list:
            clus = clus[0]
            sq_cents = []
            ref_cents = []
            i_list = [p[1][0] for p in clus]
            for point in clus:
                i, j = point[1]
                if i_list.count(i) == 1:
                    square = squares_raw[i]
                    sq_cent = np.mean(square, axis=0)
                    ref_cent = reference[j]
                    sq_cents.append(sq_cent)
                    ref_cents.append(ref_cent)

            if len(sq_cents) < 4:
                raise MacbethError(
                    '\nWARNING: No macbeth chart found!'
                    '\nNot enough squares found'
                    '\nPossible problems:\n'
                    '- Macbeth chart is occluded\n'
                    '- Macbeth chart is too dark or bright\n'
                )

            ref_cents = np.array(ref_cents)
            sq_cents = np.array(sq_cents)
            h_mat, mask = cv2.findHomography(ref_cents, sq_cents)
            if 'None' in str(type(h_mat)):
                raise MacbethError('\nERROR\n')

            mac_fit = cv2.perspectiveTransform(mac_norm, h_mat)
            mac_cen_fit = cv2.perspectiveTransform(np.array([reference]), h_mat)
            ref_mat = cv2.getPerspectiveTransform(mac_fit, np.array([ref_corns]))
            map_to_ref = cv2.warpPerspective(original_bw, ref_mat, (ref_w, ref_h))
            a = 125 / np.average(map_to_ref)
            map_to_ref = cv2.convertScaleAbs(map_to_ref, alpha=a, beta=0)
            cor = correlate(map_to_ref, ref)
            if cor > max_cor:
                max_cor = cor
                best_fit = mac_fit
                best_cen_fit = mac_cen_fit
                best_ref_mat = ref_mat

            mac_fit_inv = np.array([[mac_fit[0][2], mac_fit[0][3], mac_fit[0][0], mac_fit[0][1]]])
            mac_cen_fit_inv = np.flip(mac_cen_fit, axis=1)
            ref_mat = cv2.getPerspectiveTransform(mac_fit_inv, np.array([ref_corns]))
            map_to_ref = cv2.warpPerspective(original_bw, ref_mat, (ref_w, ref_h))
            a = 125 / np.average(map_to_ref)
            map_to_ref = cv2.convertScaleAbs(map_to_ref, alpha=a, beta=0)
            cor = correlate(map_to_ref, ref)
            if cor > max_cor:
                max_cor = cor
                best_fit = mac_fit_inv
                best_cen_fit = mac_cen_fit_inv
                best_ref_mat = ref_mat

        cor_thresh = 0.6
        if max_cor < cor_thresh:
            raise MacbethError(
                '\nWARNING: Correlation too low'
                '\nPossible problems:\n'
                '- Bad lighting conditions\n'
                '- Macbeth chart is occluded\n'
                '- Background is too noisy\n'
                '- Macbeth chart is out of camera plane\n'
            )

        copy = original.copy()
        copy = cv2.resize(original, None, fx=2, fy=2)
        for point in best_fit[0]:
            point = np.array(point, np.float32)
            point = tuple(2 * np.round(point).astype(np.int32))
            cv2.circle(copy, point, 4, (255, 0, 0), -1)
        for point in best_cen_fit[0]:
            point = np.array(point, np.float32)
            point = tuple(2 * np.round(point).astype(np.int32))
            cv2.circle(copy, point, 4, (0, 0, 255), -1)
            copy = copy.copy()
            cv2.circle(copy, point, 4, (0, 0, 255), -1)

        best_map_col = cv2.warpPerspective(original, best_ref_mat, (ref_w, ref_h))
        best_map_col = cv2.resize(best_map_col, None, fx=4, fy=4)
        a = 125 / np.average(best_map_col)
        best_map_col_norm = cv2.convertScaleAbs(best_map_col, alpha=a, beta=0)

        fit_coords = (best_fit / factor, best_cen_fit / factor)

        return (max_cor, best_map_col_norm, fit_coords, success_msg)

    except MacbethError as error:
        return (0, None, None, error)
