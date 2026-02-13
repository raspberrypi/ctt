#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright 2026 Raspberry Pi
#
# Pretty print a Raspberry Pi tuning config JSON structure in
# version 2.0 and later formats.

import json
import textwrap
from typing import Any


class Encoder(json.JSONEncoder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.indentation_level = 0
        self.hard_break = 120
        self.custom_elems: dict[str, int] = {
            'weights': 15,
            'table': 16,
            'luminance_lut': 16,
            'ct_curve': 3,
            'ccm': 3,
            'lut_rx': 9,
            'lut_bx': 9,
            'lut_by': 9,
            'lut_ry': 9,
            'gamma_curve': 2,
            'y_target': 2,
            'prior': 2,
            'tonemap': 2,
        }

    def encode(self, o: Any, node_key: str | None = None) -> str:
        if isinstance(o, (list, tuple)):
            if not any(isinstance(el, (list, tuple, dict)) for el in o):
                s = ', '.join(json.dumps(el) for el in o)
                if node_key in self.custom_elems:
                    self.indentation_level += 1
                    sl = s.split(', ')
                    num = self.custom_elems[node_key]
                    chunk = [self.indent_str + ', '.join(sl[x : x + num]) for x in range(0, len(sl), num)]
                    t = ',\n'.join(chunk)
                    self.indentation_level -= 1
                    output = f'\n{self.indent_str}[\n{t}\n{self.indent_str}]'
                elif len(s) > self.hard_break - len(self.indent_str):
                    self.indentation_level += 1
                    t = textwrap.fill(
                        s,
                        self.hard_break,
                        break_long_words=False,
                        initial_indent=self.indent_str,
                        subsequent_indent=self.indent_str,
                    )
                    self.indentation_level -= 1
                    output = f'\n{self.indent_str}[\n{t}\n{self.indent_str}]'
                else:
                    output = f' [ {s} ]'
                return output
            else:
                self.indentation_level += 1
                output = [self.indent_str + self.encode(el) for el in o]
                self.indentation_level -= 1
                output = ',\n'.join(output)
                return f' [\n{output}\n{self.indent_str}]'

        elif isinstance(o, dict):
            self.indentation_level += 1
            output = []
            for k, v in o.items():
                if isinstance(v, dict) and len(v) == 0:
                    output.append(self.indent_str + f'{json.dumps(k)}: {{ }}')
                else:
                    sep = f'\n{self.indent_str}' if isinstance(v, dict) else ''
                    output.append(self.indent_str + f'{json.dumps(k)}:{sep}{self.encode(v, k)}')
            output = ',\n'.join(output)
            self.indentation_level -= 1
            return f'{{\n{output}\n{self.indent_str}}}'

        else:
            return f' {json.dumps(o)}'

    @property
    def indent_str(self) -> str:
        return ' ' * self.indentation_level * self.indent

    def iterencode(self, o: Any, **kwargs: Any) -> str:
        return self.encode(o)


def pretty_print(in_json: dict, custom_elems: dict[str, int] | None = None) -> str:
    if custom_elems is None:
        custom_elems = {}

    if 'version' not in in_json or 'target' not in in_json or 'algorithms' not in in_json or in_json['version'] < 2.0:
        raise RuntimeError('Incompatible JSON dictionary has been provided')

    encoder = Encoder(indent=4, sort_keys=False)
    encoder.custom_elems |= custom_elems
    return encoder.encode(in_json)
