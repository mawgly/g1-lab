"""FixStand pose override for g1_ctrl's config.yaml (run.sh --fixstand-wrist).

Usage: python fixstand.py <config.yaml> <out.yaml> <SDK joint index>=<rad> [...]

g1_ctrl's FixStand interpolates from the current command to FSM.FixStand.qs[-1] (29 angles, SDK order) over
FSM.FixStand.ts. Only the given entries of that pose are rewritten, in the text itself (the rest of the file
stays byte for byte), and the result is checked with yaml: the same document except those entries.
"""

import copy
import re
import sys
from pathlib import Path

import yaml

J = 29
NUMBER = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def pose_span(text):
    """Start and end of the last inner list of the only `qs:` value: the pose FixStand moves to."""
    keys = [m.end() for m in re.finditer(r'^\s*qs:', text, re.M)]
    if len(keys) != 1:
        raise SystemExit(f'fixstand.py: expected one qs: key, found {len(keys)}')
    depth, start, last = 0, None, None
    for i in range(text.index('[', keys[0]), len(text)):
        if text[i] == '[':
            depth += 1
            if depth == 2:
                start = i
        elif text[i] == ']':
            depth -= 1
            if depth == 1:
                last = (start, i)
            elif depth == 0:
                break
    if last is None:
        raise SystemExit('fixstand.py: qs has no pose list')
    return last


def override(text, values):
    """text with the numbers of the FixStand pose at the given indices replaced by the given literals."""
    start, end = pose_span(text)
    tokens = list(NUMBER.finditer(text, start, end))
    if len(tokens) != J:
        raise SystemExit(f'fixstand.py: the FixStand pose has {len(tokens)} numbers, not {J}')
    out, pos = [], 0
    for k, m in enumerate(tokens):
        if k in values:
            out += [text[pos:m.start()], values[k]]
            pos = m.end()
    return ''.join(out + [text[pos:]])


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    source, target, *pairs = sys.argv[1:]
    values = {}
    for pair in pairs:
        joint, value = pair.split('=')
        float(value)  # a number, written as given
        if not 0 <= int(joint) < J:
            raise SystemExit(f'fixstand.py: joint {joint} is not an SDK index 0..{J - 1}')
        values[int(joint)] = value
    text = Path(source).read_text(encoding='utf-8')
    new = override(text, values)
    before, after = yaml.safe_load(text), yaml.safe_load(new)
    expected = copy.deepcopy(before)
    for joint, value in values.items():
        expected['FSM']['FixStand']['qs'][-1][joint] = float(value)
    if after != expected:
        raise SystemExit('fixstand.py: the edited file differs from the original in more than the pose entries')
    Path(target).write_text(new, encoding='utf-8')
    pose = before['FSM']['FixStand']['qs'][-1]
    for joint, value in sorted(values.items()):
        print(f'FixStand qs[-1][{joint}]: {pose[joint]} -> {value}')


if __name__ == '__main__':
    main()
