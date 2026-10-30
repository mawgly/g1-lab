"""The G1's floor-contact groups for the world model's contact-change windows (moved from scripts/lab/wm/collect.py,
same numbers): per group a bit whether a geom of it touches the floor (contact distance ≤ 0) and the summed normal force
of those contacts."""
import numpy as np

CONTACT_GROUPS = ('hand_left', 'hand_right', 'knee_left', 'knee_right', 'foot_left', 'foot_right', 'other')
# Body name parts of the groups (the rest of the robot is 'other'): hands are the wrist links (with hand_collision the
# hand capsules hang on the wrist_yaw links), knees the knee links, feet the ankle links (sole spheres, foot capsules).
GROUP_PARTS = (('wrist', 'hand'), ('knee',), ('ankle',))


def contact_groups(model, robot_geoms, groups=CONTACT_GROUPS, parts=GROUP_PARTS):
    """(ngeom,) int64 each robot geom's group index (−1 for the rest), by its body's name: 2·family + side for a left_
    or right_ body whose name holds one of the family's parts, else the last group."""
    index = np.full(model.ngeom, -1, dtype=np.int64)
    for geom in np.flatnonzero(robot_geoms):
        name = model.body(int(model.geom_bodyid[geom])).name
        side = 0 if name.startswith('left') else 1
        index[geom] = len(groups) - 1
        for family, names in enumerate(parts):
            if any(part in name for part in names) and (name.startswith('left') or name.startswith('right')):
                index[geom] = 2 * family + side
                break
    return index


def bits(touching):
    """(…,) uint8 of (…, groups) bool: bit g for group g."""
    return (touching.astype(np.uint8) << np.arange(touching.shape[-1], dtype=np.uint8)).sum(-1).astype(np.uint8)


def floor_geom(model):
    """The floor: the model's one plane geom; ValueError unless there is exactly one."""
    import mujoco
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if len(planes) != 1:
        raise ValueError(f'модель мира: ожидалась одна плоскость пола, в модели {len(planes)}')
    return int(planes[0])


def cpu_contacts(env, groups, floor=None, count=len(CONTACT_GROUPS)):
    """(touching (count,) bool, force (count,) N) of a CPU task now: its floor contacts per group; floor: the floor
    geom (None: env.floor, the getting-up task's)."""
    import mujoco
    touching, force = np.zeros(count, dtype=bool), np.zeros(count)
    model, data = env.model, env.data
    floor = env.floor if floor is None else floor
    result = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        first, second = int(contact.geom1), int(contact.geom2)
        if (first == floor) == (second == floor):
            continue
        group = groups[second if first == floor else first]
        if group < 0:
            continue
        mujoco.mj_contactForce(model, data, i, result)
        force[group] += abs(result[0])
        touching[group] |= contact.dist <= 0
    return touching, force


# lock_wrists' joints (g1_tensor_math.LOCKED_WRIST_PARTS: wrist pitch and yaw) in the G1's joint order; their motors hold
# home (angle spread 0.002 rad, speed 0.12-0.14 rad/s in candidate 8's data, the other joints 0.1-0.5 rad, 0.5-2.5
# rad/s), so their passive jitter, which the policy cannot move, gave 69 % of the «разброс» (fit-2x-spread 27.10)
LOCKED_WRISTS = (20, 21, 27, 28)
