"""unitree_mujoco's G1 with loads in the hands, as task_config arm_targets trains them (iteration 40):
hand_load.py <unitree_robots/g1 dir> <kg>[,<kg>] [--capsules]   (one number for both hands, or left,right)

Writes g1_29dof_load.xml (g1_29dof.xml with each wrist_yaw_link's inertial carrying its load at the palm: mass m + L,
COM (m·c + L·p) / (m + L), p = PALM in the link's frame, the inertia and its frame kept, exactly as the training's
g1_tensor_math.hand_load_mass does; a load of 0 leaves the link as it is) and scene_29dof_load.xml (scene_29dof.xml
including it) next to the originals, so the meshes resolve as theirs. run.sh --hand-load puts the second into
unitree_mujoco's config.yaml for one run. --capsules (run.sh --hand-capsules, 29.10) also gives each wrist_yaw_link
Unitree's hand capsule as task_config hand_collision trains it (humanoid_sim.tasks.g1_velocity HAND_CAPSULES): in
unitree_mujoco's model the rubber hand is for show (contype 0), so a policy that leans on its palms, as the getting-up
trackers do, rests on the wrist mesh there. Plain Python: WSL's venv has no humanoid_sim dependencies."""
import re
import sys
from pathlib import Path

PALM = (0.11, 0.0, 0.0)  # m, humanoid_sim.tasks.g1_tensor_math.HAND_LOAD_POSITION
BODIES = ('left_wrist_yaw_link', 'right_wrist_yaw_link')
ROBOT, SCENE = 'g1_29dof.xml', 'scene_29dof.xml'
LOADED_ROBOT, LOADED_SCENE = 'g1_29dof_load.xml', 'scene_29dof_load.xml'
# humanoid_sim.tasks.g1_velocity HAND_CAPSULES / HAND_CAPSULE_RADIUS / HAND_CAPSULE_CONTACT, Unitree's
# left_hand_collision / right_hand_collision (contype = conaffinity = 1, condim 1, priority 0, massless)
CAPSULES = {'left_wrist_yaw_link': (.07, 0., 0., .15, -.02, 0.), 'right_wrist_yaw_link': (.07, 0., 0., .15, .02, 0.)}
CAPSULE_RADIUS = .035


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


def with_capsules(text):
    """g1_29dof.xml's text with CAPSULES added to their bodies, right after each body's first inertial."""
    for body, fromto in CAPSULES.items():
        start = text.index(f'<body name="{body}"')
        match = re.compile(r'<inertial\b[^>]*/>', re.S).search(text, start)
        name = body.split('_')[0] + '_hand_collision'
        geom = (f'\n<geom name="{name}" type="capsule" size="{CAPSULE_RADIUS:g}" fromto="'
                + ' '.join(f'{v:g}' for v in fromto)
                + '" contype="1" conaffinity="1" condim="1" priority="0" group="3" density="0" rgba="0.2 0.6 0.2 0.2"/>')
        text = text[:match.end()] + geom + text[match.end():]
    return text


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    capsules = '--capsules' in argv
    argv = [a for a in argv if a != '--capsules']
    if len(argv) != 2:
        raise SystemExit(__doc__)
    folder, loads = Path(argv[0]), loads_of(argv[1])
    robot = loaded((folder / ROBOT).read_text(encoding='utf-8'), loads)
    (folder / LOADED_ROBOT).write_text(with_capsules(robot) if capsules else robot, encoding='utf-8')
    scene = (folder / SCENE).read_text(encoding='utf-8')
    if scene.count(f'file="{ROBOT}"') != 1:
        raise SystemExit(f'{folder / SCENE} does not include {ROBOT} once')
    (folder / LOADED_SCENE).write_text(scene.replace(f'file="{ROBOT}"', f'file="{LOADED_ROBOT}"'), encoding='utf-8')
    print(f'hand loads {loads[0]:g} / {loads[1]:g} kg at the palm {PALM}'
          + (', Unitree hand capsules' if capsules else '') + f': {folder / LOADED_SCENE}')


if __name__ == '__main__':
    main()
