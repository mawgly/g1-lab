"""unitree_mujoco's G1 with loads in the hands, as task_config arm_targets trains them (iteration 40):
hand_load.py <unitree_robots/g1 dir> <kg>[,<kg>]   (one number for both hands, or left,right)

Writes g1_29dof_load.xml (g1_29dof.xml with each wrist_yaw_link's inertial carrying its load at the palm: mass m + L,
COM (m·c + L·p) / (m + L), p = PALM in the link's frame, the inertia and its frame kept, exactly as the training's
g1_tensor_math.hand_load_mass does; a load of 0 leaves the link as it is) and scene_29dof_load.xml (scene_29dof.xml
including it) next to the originals, so the meshes resolve as theirs. run.sh --hand-load puts the second into
unitree_mujoco's config.yaml for one run. Plain Python: WSL's venv has no humanoid_sim dependencies."""
import re
import sys
from pathlib import Path

PALM = (0.11, 0.0, 0.0)  # m, humanoid_sim.tasks.g1_tensor_math.HAND_LOAD_POSITION
BODIES = ('left_wrist_yaw_link', 'right_wrist_yaw_link')
ROBOT, SCENE = 'g1_29dof.xml', 'scene_29dof.xml'
LOADED_ROBOT, LOADED_SCENE = 'g1_29dof_load.xml', 'scene_29dof_load.xml'


def loads_of(text):
    """(left, right) kg from '<kg>' or '<left>,<right>', each finite and non-negative."""
    values = [float(v) for v in text.split(',')]
    if len(values) == 1:
        values *= 2
    if len(values) != 2 or any(not v >= 0 or v == float('inf') for v in values):
        raise ValueError(f'loads: one or two non-negative numbers (kg), not {text!r}')
    return tuple(values)


def with_load(mass, com, load):
    """(mass, COM) of a body of `mass` with its COM `com` (3,) carrying `load` kg at PALM."""
    total = mass + load
    return total, tuple((mass * c + load * p) / total for c, p in zip(com, PALM))


def loaded(text, loads):
    """g1_29dof.xml's text with BODIES' first inertial carrying loads (left, right)."""
    for body, load in zip(BODIES, loads):
        start = text.index(f'<body name="{body}"')
        match = re.compile(r'<inertial\b[^>]*/>', re.S).search(text, start)
        element = match.group(0)
        pos = [float(v) for v in re.search(r'pos="([^"]+)"', element).group(1).split()]
        mass = float(re.search(r'mass="([^"]+)"', element).group(1))
        if not load:
            continue
        total, com = with_load(mass, pos, load)
        element = re.sub(r'pos="[^"]+"', 'pos="' + ' '.join(f'{v:.9g}' for v in com) + '"', element, count=1)
        element = re.sub(r'mass="[^"]+"', f'mass="{total:.9g}"', element, count=1)
        text = text[:match.start()] + element + text[match.end():]
    return text


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    folder, loads = Path(argv[0]), loads_of(argv[1])
    (folder / LOADED_ROBOT).write_text(loaded((folder / ROBOT).read_text(encoding='utf-8'), loads), encoding='utf-8')
    scene = (folder / SCENE).read_text(encoding='utf-8')
    if scene.count(f'file="{ROBOT}"') != 1:
        raise SystemExit(f'{folder / SCENE} does not include {ROBOT} once')
    (folder / LOADED_SCENE).write_text(scene.replace(f'file="{ROBOT}"', f'file="{LOADED_ROBOT}"'), encoding='utf-8')
    print(f'hand loads {loads[0]:g} / {loads[1]:g} kg at the palm {PALM}: {folder / LOADED_SCENE}')


if __name__ == '__main__':
    main()
