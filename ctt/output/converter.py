#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright 2026 Raspberry Pi
#
# Convert tuning file between vc4 <-> pisp formats

import json

import numpy as np

from ..platforms import pisp as pisp_mod
from ..platforms import vc4 as vc4_mod
from .json_formatter import pretty_print


def _load_template(platform_mod: object) -> dict:
    config = platform_mod.get_config()
    with open(str(config.default_template)) as f:
        return json.load(f)


def interp_2d(in_ls: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    out_ls = np.zeros((dst_h, dst_w))
    for i in range(src_h):
        out_ls[i] = np.interp(np.linspace(0, dst_w - 1, dst_w), np.linspace(0, dst_w - 1, src_w), in_ls[i])
    for i in range(dst_w):
        out_ls[:, i] = np.interp(np.linspace(0, dst_h - 1, dst_h), np.linspace(0, dst_h - 1, src_h), out_ls[:src_h, i])
    return out_ls


def convert_target(in_json: dict, target: str) -> dict:
    grid_size_pisp = pisp_mod.get_config().grid_size
    grid_size_vc4 = vc4_mod.get_config().grid_size
    json_template_pisp = _load_template(pisp_mod)
    json_template_vc4 = _load_template(vc4_mod)

    src_w, src_h = grid_size_pisp if target == 'vc4' else grid_size_vc4
    dst_w, dst_h = grid_size_vc4 if target == 'vc4' else grid_size_pisp
    json_template = json_template_vc4 if target == 'vc4' else json_template_pisp

    alsc = next(algo for algo in in_json['algorithms'] if 'rpi.alsc' in algo)['rpi.alsc']
    for colour in ['calibrations_Cr', 'calibrations_Cb']:
        if colour not in alsc:
            continue
        for temperature in alsc[colour]:
            in_ls = np.reshape(temperature['table'], (src_h, src_w))
            out_ls = interp_2d(in_ls, src_w, src_h, dst_w, dst_h)
            temperature['table'] = np.round(out_ls.flatten(), 3).tolist()

    if 'luminance_lut' in alsc:
        in_ls = np.reshape(alsc['luminance_lut'], (src_h, src_w))
        out_ls = interp_2d(in_ls, src_w, src_h, dst_w, dst_h)
        alsc['luminance_lut'] = np.round(out_ls.flatten(), 3).tolist()

    for i, algo in enumerate(in_json['algorithms']):
        if list(algo.keys())[0] == 'rpi.sdn':
            in_json['algorithms'][i] = {
                'rpi.denoise': json_template['rpi.sdn'] if target == 'vc4' else json_template['rpi.denoise']
            }
            break

    agc = next(algo for algo in in_json['algorithms'] if 'rpi.agc' in algo)['rpi.agc']
    if 'channels' in agc:
        for i, channel in enumerate(agc['channels']):
            target_agc_metering = json_template['rpi.agc']['channels'][i]['metering_modes']
            for mode, v in channel['metering_modes'].items():
                v['weights'] = target_agc_metering[mode]['weights']
    else:
        for mode, v in agc['metering_modes'].items():
            target_agc_metering = json_template['rpi.agc']['channels'][0]['metering_modes']
            v['weights'] = target_agc_metering[mode]['weights']

    if target == 'pisp':
        for i, algo in enumerate(in_json['algorithms']):
            if list(algo.keys())[0] == 'rpi.hdr':
                in_json['algorithms'][i] = {'rpi.hdr': json_template['rpi.hdr']}

    return in_json


def convert_v2(in_json: dict, target: str) -> str:
    grid_size_pisp = pisp_mod.get_config().grid_size
    grid_size_vc4 = vc4_mod.get_config().grid_size

    if 'version' in in_json and in_json['version'] == 1.0:
        converted = {
            'version': 2.0,
            'target': target,
            'algorithms': [{algo: config} for algo, config in in_json.items()],
        }
    else:
        converted = in_json

    if converted['target'] != target:
        converted = convert_target(converted, target)
        converted['target'] = target

    grid_size = grid_size_vc4[0] if target == 'vc4' else grid_size_pisp[0]
    return pretty_print(converted, custom_elems={'table': grid_size, 'luminance_lut': grid_size})
