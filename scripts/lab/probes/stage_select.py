"""CPU only. stage_select.py <stage dir | export dir> [<stage>…] [--fresh] [--workers 8] [--out x.json]

The stage selection rule of iteration 18 (the critic, 23.10; his data in lab-critic/it18), its pushes replaced in
iteration 20 (the critic, 23.10; lab-critic/it20: sweep.py, analyze.py, stand8*.json, walk8*.json). Stages of one
branch are independent draws (neighbours correlate 0.03-0.5 in drift, 0.05 in pushes), so a stage is judged on its
own:
  1. the conditions written down in advance (goal_check; not here);
  2. step 2: the stage in 7 physics variants through the Unitree bridge copy (stand_turn.run): ours and ours +6 ms,
     motor friction x0.25 and x0.25 +6 ms, no motor model, Unitree and Unitree +6 ms (+6 ms is the bridge: the PD on
     a 2 ms older state, the target 3 substeps late). Runs: stand_turn's 'bridge' group (40: ours and Unitree +6 ms,
     regular and real timing windows, our reset starts), walk 0.5 / walk 1.0 / side step for the other five (15).
     Passes with 0 falls, the worst |yaw rate| < 0.05 rad/s walking 0.5 and 1.0 m/s and stepping left and right, the
     stop right after a side step moving < 0.3 m. Then pushes in the bridge copy (bridge_kick.run: Unitree physics,
     PD on a 2 ms older state, 6 ms delay, a force M*dv/0.1 s on the pelvis), 8 directions 45 degrees apart from the
     pelvis heading: standing, 10 moments (ticks 0-457 past 3 s after the band, 0-0.91 s) x dv 0.8 / 0.9 / 1.0 /
     1.1 / 1.25 / 1.5 m/s (480 runs), and walking 0.5 m/s, 3 s into the walk, 6 phases of the 0.6 s gait cycle x dv
     0.75 / 1.0 / 1.25 (144). They pass if standing up to 1.25 m/s at most 12 of 400 fall (3 %), no cell direction x
     dv up to 1.25 has 4 falls of 10, under half of each direction falls at 1.25 (1.5 is printed, not judged), and
     walking at most 1 of 144 falls. Why (the critic): the rule before, 16 pushes of one dv (4 directions x 4
     moments 0-0.23 s), was ~8 distinct states (3 of the 4 moments left the same pelvis speed within 0.01), and the
     outcome is not monotonic in dv (a middling push gets a hesitant half step and a fall, a strong one a decisive
     step): 5968920576 "failed" 1.1 (2 of 16) holding 1.25 (0 of 16), and candidate 4's hole forward (1.0: 6 of
     10) comes from moments 0.3-0.9 s the old ones did not reach. Also printed, with no limit: the turn over the 30 s
     stand of the bridge copy (the 'bridge' group's stand30 run, Unitree +6 ms: the critic's stand30.py um6; judged
     since iteration 29 for stages with the height command, below), and
     since iteration 23 whether the feet pass through each other (foot_overlap.py: Unitree's foot capsules, the share
     of control steps with an overlap while both soles are below 2 cm, side steps ±0.5 / ±1 m/s in our physics and
     sideways pushes 1.0 / 1.25 m/s in the bridge copy; 20 more runs).
     Since iteration 24 (the critic, lab-critic/it24/shock.py, now foot_shock.py) the pushes are judged in the bridge
     copy with the feet colliding as in Unitree's training model ('Unitree +6 мс + капсулы стоп': um_probe physics
     'um_caps', task_config foot_collision's capsules with condim 3, friction 0.6, priority 1): on the robot a graze
     is a trip, and the stages of iterations 22-23 fall in 40-80 % of sideways pushes with the capsules, none without.
     Same rule, same pushes; the same 624 without the capsules ('Unitree +6 мс без капсул', the pushes of the rule
     before, comparable with earlier outputs: summary 'pushes_sim2sim') are printed as the sim2sim forecast
     (unitree_mujoco's feet pass through each other), not judged. Also judged there: side steps at ±1 m/s (stand 5
     s, left 10 s, stop 3 s, right 10 s, stop; the bridge procedure and three reset starts), 0 falls; their sideways
     speed, drift and stop move are printed.
     Since iteration 27 (the critic, lab-critic/it27/side05.py, side_grid.py) every side step of the 7 variants (left
     then right at ±0.3 m/s) runs again at ±0.5 m/s, plan 'side05' with the same segments, start and timing (the
     bridge group's 18 and one for each of the other five: 23 runs), judged as at ±0.3 (worst |yaw rate| < 0.05, the
     stop right after it < 0.3 m) and printed in columns of its own: the floor works at ±0.5, the stick's limit in the
     export's deploy.yaml, while step 2 judged drift and stop at ±0.3 only, and at ±0.5 in the bridge copy 7141294080
     steps left with -0.057, 7121240064 right with +0.062 (both pass at ±0.3), candidate 7 (exp-c362df8e578148d0 @
     6368231424) -0.040. Under it 7121240064 fails step 2: stepping right +0.063 in the bridge copy (reset starts),
     the stop after stepping left 0.30-0.35 m in the four variants without +6 ms (with it 0.11-0.17).
     Since iteration 29 (the critic, lab-critic/it29c: cwalk.py, yawc.py, screen.json, cstand.json) a stage with the
     height command (task_config height_command) is judged in the bridge copy also on what the crouch changes and step
     2 did not see: the walk at 0.5 m/s at the height 0.70 (stand_turn plan crouch_walk05: stand 5 s at 0.78, walk 10 s
     at 0.70 as g1_ctrl's height patch gives with the stick down, stop 5 s at 0.78), |yaw rate| < 0.03, and the turn
     over the 30 s stand at 0.78 (the bridge group's stand30), |.| <= 10 degrees; the turn over 30 s crouched at 0.74
     and 0.64 (crouch_stand74 / 64, the height from the Velocity entry on) is printed only, a fall there too (3 runs
     more). 7701331968 passed step 2 (every drift <= 0.036) yet in the Unitree bridge walks crouched with
     -0.049...-0.052 (the copy -0.051, our physics with the delay does not show it) and turns -16...-18 degrees in the
     30 s stand (the copy -15.8, candidate 8 -8); crouched it turns +25.5 at 0.74, 7741440000 -14.8 at 0.64. Of the
     branch's 37 stages (7601-7962) 14 walk crouched below 0.03, 23 turn within 10 degrees standing, 10 do both. Printed
     in a table of their own (a row each, the limit last), their reasons after the others ('команда высоты: ...'), and
     after every stage a table across the stages with the height command (step 2 without them and with them). Stages
     without the height command keep the rule before, the 30 s turn printed only: the rule that chose candidate 8 is
     unchanged.
     Since iteration 42 (the user's request of 25.10, chatter.py) step 2 also judges torque chattering, on the CPU in
     our physics: the deterministic policy with the CUDA task's sensor noise and encoder bias and its training delay,
     six scenarios (standing, 0.5 / 1 m/s, stop-start, side steps ±0.3 m/s) x 3 seeds, 10 s each, and the same runs of
     Unitree v0 (18 more, once per training delay among the stages). Judged ('дрожание моментов'): the worst driven
     joint's motor torque above 10 Hz (RMS, N m) and PD target above 5 Hz (RMS, rad) at most 1.5 x v0's, and no fall;
     printed in a line of its own with the shares, the torque RMS over the motor peak and the mean |Δa|. Candidate 8
     (exp-28691193718e4d0e @ 7221510144) 4.57 against 4.08 N m, 0.135 against 0.112 rad: passes. An arm_targets stage
     also runs the carry pose with 1 kg per hand (3 runs, chatter.py prints it, not judged). ~2.3 s of a process a run.
     1368 runs a stage, 1371 with the height command (1350 / 1353 before iteration 42, 1327 before iteration 27, 699
     before iteration 24), and the reference's 18;
  3. --fresh: one check of the winner on cases step 2 did not see, in the bridge copy (Unitree +6 ms): other timing
     windows (side step at offsets 5-61, walks at 13; step 2 has 0 and 0-53), delay 4 and 8 ms, stop-start, a right
     step alone (19 runs, no pushes since iteration 20: step 2 covers them). Same limits; the 4 / 8 ms runs and
     stop-start are judged on falls and the stop only, their drift is printed (the delay itself moves it: 5631836160
     stepping left -0.037 at 6 ms, -0.051 at 8 ms).
An export_g1 folder is judged as exported (export_g1.ExportedPolicy: its policy.onnx with the stand trim, in the
environment of its stage), as in stand_turn. Prints per stage a table by physics variant (the ±0.5 m/s side step's
drift and stop in columns of their own), the 30 s turn, the height checks (stages with the height command), the
standing pushes by direction x dv, the walking pushes and the verdict, then the height checks across stages and the
share of stages that pass (the measure of a recipe). Every run is deterministic (seed 73000): the same stage gives the
same numbers, run for run equal to the critic's files (the 624 pushes of each stage below: falls, peak speed and
lowest pelvis bit for bit as in his stand8*.json / walk8*.json; the 30 s turn as his stand30*.json um6; the ±0.5 side
steps as his side05.json; the crouched walk and stands of 7701331968 and 7741440000, and their 30 s turn, as his
it29c screen.json cwalk / stand30 and cstand.json cstand0.74 / 0.64 um). 8 workers (23.10, the trainer on the GPU, 24
logical cores): ~4 min a stage for step 2 (679 runs, 624 of them pushes, ~1800 s of processes), ~12 s for step 3 (4
stages with --fresh: 990 s). Since iteration 27 (23.10, 4 workers, the trainer on the GPU): step 2 of 7121240064 in
423 s, 1545 s of processes for 1350 runs, the ±0.5 side steps 74 s of them (+5 %). Since iteration 29 (24.10, 2
workers, the trainer on the GPU): step 2 of 7701331968 and 7741440000 in 1614 s, 1481 / 1484 s of processes for 1353
runs a stage, the three height runs ~6 s of them (+0.4 %).
Iteration 29 (exp-04f74e3838184b82), step 2: 7701331968 fails on the height checks only (crouched walk -0.051, the 30
s turn -15.8 deg; crouched +25.5 at 0.74, +10.0 at 0.64), 7741440000 passes (-0.023, -3.5; +2.2, -14.8).
The stages of the critic's check (iteration 20, exp-8ce0cbd70d0a4f1d, and candidate 4, exp-902b8958ee504780 @
5631836160), step 2:
  5968920576 standing 8 of 400 (worst backwards 1.0: 3 of 10), walking 0 of 144, turn 30 s +4.5 deg: passes;
  5631836160 standing 45 of 400 (forward 1.0: 6 of 10; 1.25 forward and backwards 10 of 10, right 8 of 10), walking
  8 of 144, turn +21.9 deg: fails on the pushes only;
  5988974592, 6039109632 0 of 400 and 0 of 144, turn +18.2 / +14.6 deg: pass (step 3: 5988974592 fails, the stop
  after a side step 0.35 m in another timing window).
Iteration 18 (the critic's runs of 25 stages, the pushes before): step 2 passed 5631836160 only.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIM = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bridge_kick as kick  # noqa: E402
import chatter as ch  # noqa: E402
import foot_overlap as feet  # noqa: E402
import stand_turn as st  # noqa: E402
import um_probe as um  # noqa: E402

STEPPER = kick.PushStepper.__bases__[0]
um.Stepper = STEPPER  # bridge_kick puts its push stepper in on import; only push runs use it (work)
DIRS8 = {'a%03d' % (45 * i): (math.cos(math.radians(45 * i)), math.sin(math.radians(45 * i))) for i in range(8)}
kick.DIRS = DIRS8  # the critic's 8 (lab-critic/it20/sweep.py), in the pelvis heading: 0 = forward, 90 = left
DIR_LABELS = dict(a000='вперёд', a045='вперёд-влево', a090='влево', a135='назад-влево', a180='назад',
                  a225='назад-вправо', a270='вправо', a315='вперёд-вправо')

DRIFT_LIMIT = .05  # rad/s, worst |yaw rate| walking or stepping sideways with no turn command
STOP_LIMIT = .3  # m, the move in the stop right after a side step (the falls of iteration 18 came 1.5-2.6 s into it)
STAND_MAGS = (.8, .9, 1., 1.1, 1.25, 1.5)  # m/s, dv of the standing pushes
STAND_OFFSETS = (0, 37, 71, 113, 151, 197, 251, 307, 379, 457)  # ticks of 2 ms past 3 s after the band
WALK_MAGS = (.75, 1., 1.25)  # m/s, dv of the pushes walking 0.5 m/s
WALK_OFFSETS = (0, 50, 100, 150, 200, 250)  # ticks past 3 s into the walk: 6 phases of the 0.6 s gait cycle
PUSHES = (('stand', STAND_MAGS, STAND_OFFSETS), ('walk', WALK_MAGS, WALK_OFFSETS))
JUDGED_MAG = 1.25  # standing pushes up to this dv are judged; stronger ones are printed
STAND_FALLS = 12  # most falls of the 400 judged standing pushes (3 %)
CELL_FALLS = 4  # falls of the 10 moments in one direction x dv cell (dv < JUDGED_MAG: a systematic hole in the
# moderate pushes; at JUDGED_MAG itself, the edge of the margin, HALF judges a direction) that fail a stage
HALF = .5  # share of falls at JUDGED_MAG in one direction that fails a stage
WALK_FALLS = 1  # most falls of the 144 walking pushes
SIX_MS = dict(stale=True, delay=3)
VARIANTS = (  # name, label, stand_turn case fields
    ('ours', 'наша', dict(physics='ours', stale=False, delay=0)),
    ('ours_6ms', 'наша +6 мс', dict(physics='ours', **SIX_MS)),
    ('fric', 'трение моторов x0.25', dict(physics='ours', extra='fric0.25', stale=False, delay=0)),
    ('fric_6ms', 'трение x0.25 +6 мс', dict(physics='ours', extra='fric0.25', **SIX_MS)),
    ('nomotor', 'без моторной модели', dict(physics='nomotor', stale=False, delay=0)),
    ('um', 'Unitree', dict(physics='um', stale=False, delay=0)),
    ('um_6ms', 'Unitree +6 мс (связка)', dict(physics='um', **SIX_MS)),
)
BRIDGE = ('ours_6ms', 'um_6ms')  # stand_turn's 'bridge' group runs these two
# Since iteration 27 every side step of step 2 runs again at ±0.5 m/s, the stick's limit in the export's deploy.yaml
# where the floor works (the critic, lab-critic/it27/side05.py, side_grid.py): judged at ±0.3 only, the rule let through
# drifts of -0.057 (7141294080 stepping left) and +0.062 (7121240064 right) in the bridge copy at ±0.5.
SIDE05 = 'side05'  # stand_turn plan: 'side' at ±0.5 m/s, segments 'left05' / 'right05'
PLANS = ('side', SIDE05, 'walk', 'walk1')  # the other five: side step ±0.3 and ±0.5, walk 0.5, walk 1.0
FRESH = (  # group, label, judged on drift
    ('windows', 'другие окна времени', True),
    ('right', 'шаг вправо', True),
    ('stopstart', 'стоп-старт', False),
    ('delay4', 'задержка 4 мс', False),
    ('delay8', 'задержка 8 мс', False),
)
DRIFT = (('w05', '0.5 м/с'), ('w1', '1 м/с'), ('L', 'влево'), ('R', 'вправо'))
DRIFT05 = (('L05', 'влево 0.5 м/с'), ('R05', 'вправо 0.5 м/с'))  # the ±0.5 m/s side step, step 2 only
STOPS = (('stop', ''), ('stop05', ' 0.5 м/с:'))  # row key, label: the stop right after a ±0.3 / ±0.5 m/s side step
# a side step segment of stand_turn's plans -> its drift key and the key of the stop right after it
SIDE_SEGMENTS = dict(left=('L', 'stop'), right=('R', 'stop'), left05=('L05', 'stop05'), right05=('R05', 'stop05'))
# Since iteration 24 the pushes and the ±1 m/s side check are judged in the Unitree bridge copy with the feet colliding
# as in Unitree's training model (um_probe 'um_caps': task_config foot_collision's capsules, condim 3, friction 0.6):
# on the robot a foot grazing the other is a trip, and without the capsules the feet pass through each other (the
# critic, lab-critic/it24: the same stages fall in 40-80 % of sideways pushes with them, none without). The same
# pushes without the capsules ('um', as unitree_mujoco, whose feet pass through each other: the sim2sim forecast, and
# the numbers of the rule before) are printed, not judged.
PUSH_PHYSICS = 'um_caps'
FORECAST_PHYSICS = 'um'
PHYSICS_LABELS = {PUSH_PHYSICS: 'Unitree +6 мс + капсулы стоп', FORECAST_PHYSICS: 'Unitree +6 мс без капсул'}
SIDE_CHECK = 'side1'  # stand_turn plan: stand, 1 m/s left 10 s, stop, 1 m/s right 10 s, stop; judged on falls
SIDE_CHECK_SEGMENTS = (('left1', 'влево'), ('right1', 'вправо'))
# Since iteration 29 (the critic, lab-critic/it29c: cwalk.py, yawc.py, screen.json, cstand.json) a stage with the height
# command (task_config height_command) is judged in the bridge copy (Unitree +6 ms) also on what the crouch changes and
# step 2 did not see: 7701331968 passed step 2 (every drift <= 0.036) yet in the Unitree bridge walks 0.5 m/s at the
# height 0.70 with -0.049...-0.052 (the copy -0.051; 14 of the branch's 37 stages stay below 0.03) and turns -16...-18
# deg in the 30 s stand (the copy -15.8, candidate 8 -8; 14 of 37 within +-5). Judged: the crouched walk (stand_turn
# plan crouch_walk05), |yaw rate| < CROUCH_DRIFT_LIMIT, and the 30 s stand at 0.78 (the bridge group's stand30, printed
# for every stage), |turn| <= STAND_TURN_LIMIT; for stages without the height command that stand stays printed only, so
# the rule that chose candidate 8 is unchanged. Printed: the turn over 30 s crouched at 0.74 and 0.64 (crouch_stand74
# / 64, measured nowhere before: 7701331968 +25.5 deg at 0.74, 7741440000 -14.8 at 0.64).
CROUCH_WALK = 'crouch_walk05'  # stand 5 s at 0.78, walk 0.5 m/s 10 s at 0.70, stop 5 s at 0.78
CROUCH_STANDS = ('crouch_stand74', 'crouch_stand64')  # 30 s standing at 0.74 / 0.64
CROUCH_DRIFT_LIMIT = .03  # rad/s, worst |yaw rate| of the crouched walk
STAND_TURN_LIMIT = 10.  # degrees, largest |turn| over the 30 s stand at 0.78 (stages with the height command)
CROUCH_VARIANT = 'um_6ms'  # the bridge copy: the height checks run there only


def push_jobs(stage):
    """The pushes of step 2 (lab-critic/it20/sweep.py): bridge_kick.run jobs, standing and walking 0.5 m/s, in the
    Unitree bridge copy with the foot capsules (judged) and without them (printed)."""
    return [('push', dict(name=label(stage), path=str(stage), left=False, phys=phys, dir=d, mag=m, offset=o,
                          plan=plan)) for phys in (PUSH_PHYSICS, FORECAST_PHYSICS)
            for plan, mags, offsets in PUSHES for d in DIRS8 for m in mags for o in offsets]


def side_check_jobs(stage):
    """The ±1 m/s side check of step 2 in the bridge copy with the foot capsules: the bridge procedure (FixStand on the
    band) and three of our reset starts (gait phase 0, 15, 30), as the 'bridge' group's side step."""
    fields = dict(physics=PUSH_PHYSICS, plan=SIDE_CHECK, stage=str(stage), **SIX_MS)
    return [st.case(**fields)] + [st.case(start='reset', phase=15 * k, seed=73000 + k, **fields) for k in range(3)]


def height_jobs(stage):
    """The height command checks of step 2 (iteration 29): the crouched walk and the two crouched stands, each one run
    in the bridge copy (Unitree +6 ms, the bridge procedure), as the critic's cwalk.py."""
    fields = next(fields for name, _, fields in VARIANTS if name == CROUCH_VARIANT)
    return [st.case(plan=plan, stage=str(stage), **fields) for plan in (CROUCH_WALK,) + CROUCH_STANDS]


def select_jobs(stage, height=None, arms=None):
    """Step 2 of one stage: [(kind, job)], kind 'run' (stand_turn.run case: the 7 physics variants, the bridge group's
    side steps again at ±0.5 m/s, then the side check and, with the height command, the height checks), 'push'
    (bridge_kick.run job), 'feet' (foot_overlap.run job, printed only) or 'chatter' (chatter.run job; the reference's
    runs are main's, once per delay). height: whether the stage has task_config height_command; arms: whether it has
    arm_targets (chatter's carry scenario); None: read it from the stage."""
    if height is None:
        height = st.has_height(stage)
    if arms is None:
        arms = ch.has_arm_targets(str(stage))
    bridge = st.grid(str(stage), 'bridge')
    jobs = [('run', case) for case in bridge]
    jobs += [('run', dict(case, plan=SIDE05)) for case in bridge if case['plan'] == 'side']
    for name, _, fields in VARIANTS:
        if name not in BRIDGE:
            jobs += [('run', st.case(plan=plan, stage=str(stage), **fields)) for plan in PLANS]
    jobs += [('run', case) for case in side_check_jobs(stage)]
    if height:
        jobs += [('run', case) for case in height_jobs(stage)]
    return (jobs + push_jobs(stage) + [('feet', job) for job in feet.jobs(stage)]
            + [('chatter', job) for job in ch.jobs(str(stage), arms)])


def fresh_jobs(stage):
    """Step 3 of one stage, the critic's fresh check (lab-critic/it18/hold_make.py)."""
    bridge = dict(physics='um', **SIX_MS)
    cases = [st.case(plan='side', timing='real', which=0, offset=o, **bridge) for o in (5, 17, 29, 43, 61)]
    cases += [st.case(plan='walk', timing='real', which=w, offset=13, **bridge) for w in range(6)]
    cases += [st.case(plan='walk1', timing='real', which=w, offset=13, **bridge) for w in range(2)]
    cases += [st.case(plan=plan, physics='um', stale=True, delay=d) for d in (2, 4) for plan in ('side', 'walk1')]
    cases += [st.case(plan=plan, **bridge) for plan in ('stopstart', 'sideR')]
    return [('run', dict(case, stage=str(stage))) for case in cases]


def variant_of(case):
    key = (case['physics'], case.get('extra'), bool(case.get('stale')), int(case.get('delay', 0)))
    for name, _, fields in VARIANTS:
        if key == (fields['physics'], fields.get('extra'), fields['stale'], fields['delay']):
            return name
    raise ValueError(f'not one of the 7 physics variants: {case}')


def fresh_group(case):
    if case.get('timing') == 'real':
        return 'windows'
    if case['plan'] in ('sideR', 'stopstart'):
        return 'right' if case['plan'] == 'sideR' else 'stopstart'
    if case['physics'] != 'um' or not case.get('stale') or case['delay'] not in (2, 4):
        raise ValueError(f'not a case of the fresh check: {case}')
    return 'delay%d' % (2 * case['delay'])


def measures(result):
    """One stand_turn.run result: yaw rates walking ('w05', 'w1') and stepping at ±0.3 ('L', 'R') and ±0.5 m/s ('L05',
    'R05'), speeds ('v05', 'v1'), the move in a stop right after a side step ('stop'; after ±0.5 'stop05') and in the
    stops of stop-start ('brake'). A run that fell gives none (its fall is counted)."""
    out = {key: [] for key in ('w05', 'w1', 'L', 'R', 'L05', 'R05', 'v05', 'v1', 'stop', 'stop05', 'brake')}
    if result['fell_at'] is not None:
        return out
    plan, prev = result['case']['plan'], None
    for seg in result['segs']:
        name = seg['name']
        if not seg.get('missing'):
            if plan == 'stopstart':  # 3 s walks: no drift
                if name == 'stop':
                    out['brake'].append(seg['disp'])
            elif name == 'walk':
                key = 'w1' if seg['cmd'][0] > .75 else 'w05'
                out[key].append(seg['yaw_rate'])
                out['v' + key[1:]].append(seg['fwd'])
            elif name in SIDE_SEGMENTS:
                out[SIDE_SEGMENTS[name][0]].append(seg['yaw_rate'])
            elif name == 'stop' and prev in SIDE_SEGMENTS:
                out[SIDE_SEGMENTS[prev][1]].append(seg['disp'])
        prev = name
    return out


def fall_of(result):
    c = result['case']
    real = c.get('timing') == 'real'
    where = c['plan'] + (' окно %s/%s' % (c.get('which', 0), c.get('offset', 0)) if real else '')
    where += ' старт reset' if c.get('start') == 'reset' else ''
    return where + (' %.1f с' % result['fell_at'] if result['fell_at'] >= 0 else '')


def row(results):
    """Table row of a group of stand_turn.run results: worst (largest |.|) yaw rates, largest stop moves, speeds."""
    got = {}
    for r in results:
        for key, values in measures(r).items():
            got.setdefault(key, []).extend(values)
    worst = {key: (max(got.get(key, []), key=abs) if got.get(key) else None) for key, _ in DRIFT + DRIFT05}
    return dict(runs=len(results), falls=[fall_of(r) for r in results if r['fell_at'] is not None], drift=worst,
                stop=max(got.get('stop', []), default=None), stop05=max(got.get('stop05', []), default=None),
                brake=max(got.get('brake', []), default=None),
                speed={key: (float(np.mean(got[key])) if got.get(key) else None) for key in ('v05', 'v1')})


def judge(rows, labels, drift_rows, drift=DRIFT, stops=STOPS[:1]):
    """Reasons a stage does not pass ([] = passes). rows: name -> row; drift_rows: the names judged on drift; drift,
    stops: the (key, label) of the yaw rates and side-step stops judged (step 2 adds the ±0.5 m/s side step's)."""
    reasons = []
    falls = ['%s: %s' % (labels[name], f) for name, r in rows.items() for f in r['falls']]
    if falls:
        reasons.append('падений %d (%s)' % (len(falls), '; '.join(falls)))
    for key, what in drift:
        values = [(abs(rows[name]['drift'][key]), rows[name]['drift'][key], name) for name in drift_rows
                  if rows[name]['drift'][key] is not None]
        if not values:
            reasons.append('увод %s не измерен' % what)
            continue
        size, value, name = max(values)
        if size >= DRIFT_LIMIT:
            reasons.append('увод %s %+.3f, |.| >= %.2f (%s)' % (what, value, DRIFT_LIMIT, labels[name]))
    for key, what in stops:
        moves = [(r[key], name) for name, r in rows.items() if r[key] is not None]
        if not moves:
            reasons.append('сдвиг после стопа за шагом вбок%s не измерен' % what)
        elif max(moves)[0] >= STOP_LIMIT:
            reasons.append('сдвиг после стопа за шагом вбок%s %.2f м >= %.1f (%s)' % (
                what, max(moves)[0], STOP_LIMIT, labels[max(moves)[1]]))
    return reasons


def dv(mag):
    """1.0 -> '1.0', 1.25 -> '1.25'."""
    text = '%.2f' % mag
    return text[:-1] if text.endswith('0') else text


def push_table(results, plan, mags, physics=PUSH_PHYSICS):
    """bridge_kick.run results of one plan in one physics: falls and runs by direction, one entry per dv of `mags`,
    and the falls that came before the push ('early', counted among the falls)."""
    falls, runs, early = {d: [0] * len(mags) for d in DIRS8}, {d: [0] * len(mags) for d in DIRS8}, 0
    for r in results:
        job = r['job']
        if job['plan'] != plan or job['phys'] != physics:
            continue
        k = [k for k, m in enumerate(mags) if abs(m - job['mag']) < 1e-6]
        if not k:
            raise ValueError(f'a push of dv {job["mag"]} is not one of {mags}')
        runs[job['dir']][k[0]] += 1
        falls[job['dir']][k[0]] += r['fell'] is not None
        early += bool(r.get('fell_before'))
    return dict(mags=list(mags), falls=falls, runs=runs, early=early)


def judged(table):
    """Indices of the judged dv of a standing push table and the falls and runs over them."""
    ks = [k for k, m in enumerate(table['mags']) if m <= JUDGED_MAG + 1e-9]
    return (ks, sum(table['falls'][d][k] for d in DIRS8 for k in ks),
            sum(table['runs'][d][k] for d in DIRS8 for k in ks))


def push_reasons(stand, walk):
    """Reasons the pushes of a stage do not pass ([] = pass): stand and walk are push_table()s."""
    reasons = []
    for table, what, offsets in ((stand, 'стоя', STAND_OFFSETS), (walk, 'на ходу', WALK_OFFSETS)):
        got = sum(sum(n) for n in table['runs'].values())
        if any(n != len(offsets) for runs in table['runs'].values() for n in runs):
            reasons.append('толчков %s %d, нужно %d (по %d в клетке)' % (
                what, got, len(DIRS8) * len(table['mags']) * len(offsets), len(offsets)))
    if reasons:
        return reasons
    ks, falls, runs = judged(stand)
    if falls > STAND_FALLS:
        reasons.append('толчки стоя до %s м/с: %d из %d, можно до %d' % (dv(JUDGED_MAG), falls, runs, STAND_FALLS))
    cells = ['%s %s — %d' % (DIR_LABELS[d], dv(stand['mags'][k]), stand['falls'][d][k])
             for d in DIRS8 for k in ks
             if stand['mags'][k] < JUDGED_MAG - 1e-9 and stand['falls'][d][k] >= CELL_FALLS]
    if cells:
        reasons.append('клетки стоя с %d из %d и больше: %s' % (CELL_FALLS, len(STAND_OFFSETS), ', '.join(cells)))
    top = stand['mags'].index(JUDGED_MAG)
    half = ['%s %d из %d' % (DIR_LABELS[d], stand['falls'][d][top], stand['runs'][d][top]) for d in DIRS8
            if stand['falls'][d][top] >= HALF * stand['runs'][d][top]]
    if half:
        reasons.append('толчки стоя %s м/с: падает половина и больше (%s)' % (dv(JUDGED_MAG), ', '.join(half)))
    falls = sum(sum(n) for n in walk['falls'].values())
    if falls > WALK_FALLS:
        reasons.append('толчки на ходу 0.5 м/с: %d из %d, можно до %d (%s)' % (
            falls, sum(sum(n) for n in walk['runs'].values()), WALK_FALLS, walk_where(walk)))
    return reasons


def walk_where(walk):
    """Where the walking pushes fell: 'вперёд-влево 3, влево 2'."""
    return ', '.join('%s %d' % (DIR_LABELS[d], sum(walk['falls'][d])) for d in DIRS8 if sum(walk['falls'][d]))


def stand30(runs):
    """The turn over the 30 s stand of the bridge copy (the 'bridge' group's stand30 run, Unitree +6 ms; the critic's
    stand30.py um6), degrees, printed with no limit; None if it fell or was not run."""
    for r in runs:
        if r['case']['plan'] == 'stand30' and variant_of(r['case']) == 'um_6ms':
            seg = r['segs'][0] if r['segs'] else dict(missing=True)
            ok = r['fell_at'] is None and not seg.get('missing')
            return dict(heading=seg['heading_deg'] if ok else None, fell=r['fell_at'])
    return dict(heading=None, fell=None)


def side_check(results):
    """The ±1 m/s side check from its stand_turn.run results: runs, falls, the mean sideways speed and the worst
    |yaw rate| stepping left and right, the largest move in the stops after them (printed; falls are judged)."""
    speeds, drift, stops = {key: [] for key, _ in SIDE_CHECK_SEGMENTS}, {key: [] for key, _ in SIDE_CHECK_SEGMENTS}, []
    for r in results:
        if r['fell_at'] is not None:
            continue
        prev = None
        for seg in r['segs']:
            if not seg.get('missing'):
                if seg['name'] in speeds:
                    speeds[seg['name']].append(seg['side'])
                    drift[seg['name']].append(seg['yaw_rate'])
                elif seg['name'] == 'stop' and prev in speeds:
                    stops.append(seg['disp'])
            prev = seg['name']
    return dict(runs=len(results), falls=[fall_of(r) for r in results if r['fell_at'] is not None],
                speed={key: (float(np.mean(v)) if v else None) for key, v in speeds.items()},
                drift={key: (max(v, key=abs) if v else None) for key, v in drift.items()},
                stop=max(stops, default=None))


def side_reasons(side):
    """Reasons the ±1 m/s side check does not pass ([] = passes): a fall, or no run."""
    if not side['runs']:
        return ['шаг вбок ±1 м/с не измерен']
    if side['falls']:
        return ['шаг вбок ±1 м/с (%s): падений %d из %d (%s)' % (
            PHYSICS_LABELS[PUSH_PHYSICS], len(side['falls']), side['runs'], '; '.join(side['falls']))]
    return []


def variant_label(name):
    """The label of a physics variant."""
    return next(text for key, text, _ in VARIANTS if key == name)


def crouch_height(plan, name):
    """The height command (m) of segment `name` of a stand_turn height plan."""
    return next(segment[5] for segment in st.PLANS[plan] if segment[0] == name)


def crouch_run(runs, plan, name):
    """Segment `name` of the plan's stand_turn.run result: runs, the fall ('fall_of' text) and the segment (None if
    the run fell or did not reach it)."""
    found = [r for r in runs if r['case']['plan'] == plan]
    if not found:
        return dict(runs=0, fell=None, seg=None)
    seg = next((s for s in found[0]['segs'] if s['name'] == name and not s.get('missing')), None)
    fell = None if found[0]['fell_at'] is None else fall_of(found[0])
    return dict(runs=len(found), fell=fell, seg=None if fell else seg)


def height_check(runs, turn):
    """The height command checks of step 2 (iteration 29; stages with task_config height_command) from their
    stand_turn.run results and the 30 s stand's turn (stand30()): the crouched walk's yaw rate, speed and pelvis, the
    crouched stands' turns and pelvis, and the reasons the stage fails them ([] = passes): the crouched walk fell, was
    not measured or has |yaw rate| >= CROUCH_DRIFT_LIMIT; the 30 s stand at 0.78 was not measured or turned |.| >
    STAND_TURN_LIMIT degrees. The crouched stands are printed only, their falls too."""
    where = variant_label(CROUCH_VARIANT)
    got = crouch_run(runs, CROUCH_WALK, 'walk')
    seg = got['seg'] or {}
    walk = dict(height=crouch_height(CROUCH_WALK, 'walk'), runs=got['runs'], fell=got['fell'],
                drift=seg.get('yaw_rate'), speed=seg.get('fwd'), pelvis=seg.get('pelvis'))
    stands = []
    for plan in CROUCH_STANDS:
        got = crouch_run(runs, plan, 'stand')
        seg = got['seg'] or {}
        stands.append(dict(plan=plan, height=crouch_height(plan, 'stand'), runs=got['runs'], fell=got['fell'],
                           heading=seg.get('heading_deg'), pelvis=seg.get('pelvis')))
    what = 'на ходу 0.5 м/с на %.2f' % walk['height']
    reasons = []
    if walk['fell']:
        reasons.append('падение %s (%s: %s)' % (what, where, walk['fell']))
    elif walk['drift'] is None:
        reasons.append('увод %s не измерен' % what)
    elif abs(walk['drift']) >= CROUCH_DRIFT_LIMIT:
        reasons.append('увод %s %+.3f, |.| >= %.2f (%s)' % (what, walk['drift'], CROUCH_DRIFT_LIMIT, where))
    if turn['heading'] is None:
        reasons.append('поворот стоя 30 с не измерен')
    elif abs(turn['heading']) > STAND_TURN_LIMIT:
        reasons.append('поворот стоя 30 с %+.1f°, |.| > %g° (%s)' % (turn['heading'], STAND_TURN_LIMIT, where))
    return dict(walk=walk, stands=stands, reasons=reasons)


def select_summary(runs, push_results, height=False, chatter=None):
    """Step 2 from the stage's stand_turn.run results (7 variants, the ±1 m/s side check, the height checks) and
    bridge_kick.run results (standing and walking, with the foot capsules and without). 'pushes' are the judged ones
    (since iteration 24: Unitree +6 ms with the capsules), 'pushes_sim2sim' the ones without them (the rule's numbers
    before). height: the stage has task_config height_command, so the height checks judge it ('height': their numbers
    and their own reasons, which also end 'reasons'); without it 'height' is None and the rule is the one before.
    chatter: chatter.summarize() of the stage's chatter runs against the reference's (iteration 42), judged by its
    summary's ok, its reasons before the height checks' ('дрожание моментов: ...'); None: not run, nothing judged."""
    checked = [r for r in runs if r['case']['plan'] == SIDE_CHECK]
    crouched = [r for r in runs if r['case']['plan'] in st.HEIGHT_PLANS]
    runs = [r for r in runs if r['case']['plan'] != SIDE_CHECK and r['case']['plan'] not in st.HEIGHT_PLANS]
    rows = {name: row([r for r in runs if variant_of(r['case']) == name]) for name, _, _ in VARIANTS}
    labels = {name: text for name, text, _ in VARIANTS}
    pushes = {plan: push_table(push_results, plan, mags, PUSH_PHYSICS) for plan, mags, _ in PUSHES}
    forecast = {plan: push_table(push_results, plan, mags, FORECAST_PHYSICS) for plan, mags, _ in PUSHES}
    side = side_check(checked)
    turn = stand30(runs)
    crouch = height_check(crouched, turn) if height else None
    reasons = (judge(rows, labels, [name for name, _, _ in VARIANTS], DRIFT + DRIFT05, STOPS)
               + push_reasons(pushes['stand'], pushes['walk']) + side_reasons(side))
    if chatter and not chatter['summary']['ok']:
        reasons.append('дрожание моментов: ' + '; '.join(chatter['reasons']))
    if crouch:  # their own reasons, after the rest
        reasons += crouch['reasons']
    return dict(passes=not reasons, reasons=reasons, rows=rows, pushes=pushes, pushes_sim2sim=forecast, side1=side,
                stand30=turn, height=crouch, chatter=chatter)


def fresh_summary(runs):
    """Step 3 from the fresh check's stand_turn.run results."""
    rows = {name: row([r for r in runs if fresh_group(r['case']) == name]) for name, _, _ in FRESH}
    labels = {name: text for name, text, _ in FRESH}
    reasons = judge(rows, labels, [name for name, _, judged in FRESH if judged])
    return dict(passes=not reasons, reasons=reasons, rows=rows)


def label(path):
    path = Path(path)
    if (path / 'policy.onnx').is_file():
        return path.name + ' (выгрузка)'
    return '%s @ %s' % (path.parent.name, path.name[6:].lstrip('0'))


def number(value, fmt):
    return '-' if value is None else fmt % value


def print_rows(title, summary, labels, side05=False):
    """A table of rows by name; side05 (step 2): the ±0.5 m/s side step's yaw rates and stop in columns of their own."""
    print(title)
    extra = ' %9s %10s %8s' % ('влево 0.5', 'вправо 0.5', 'стоп 0.5') if side05 else ''
    print('  %-24s %8s %8s %8s %8s %8s %8s %7s%s %12s' % (
        '', 'прогонов', 'падений', '0.5 м/с', '1 м/с', 'влево', 'вправо', 'стоп, м', extra, 'v 0.5 / 1'))
    for name, text in labels:
        r = summary['rows'][name]
        speed = '%s / %s' % (number(r['speed']['v05'], '%.2f'), number(r['speed']['v1'], '%.2f'))
        extra = ' %9s %10s %8s' % (*(number(r['drift'][key], '%+.3f') for key, _ in DRIFT05),
                                   number(r['stop05'], '%.2f')) if side05 else ''
        print('  %-24s %8d %8d %8s %8s %8s %8s %7s%s %12s' % (
            text, r['runs'], len(r['falls']), *(number(r['drift'][key], '%+.3f') for key, _ in DRIFT),
            number(r['stop'], '%.2f'), extra, speed)
              + ('' if r['brake'] is None else '  стоп-старт %.2f м' % r['brake']))


def print_push_tables(pushes, physics, note):
    """The standing pushes by direction x dv and the walking pushes of one physics (its label and a note)."""
    stand, walk = pushes['stand'], pushes['walk']
    what = '%s, %s' % (PHYSICS_LABELS[physics], note)
    print('  толчки стоя (%s): падений из %d моментов 0-%.2f с, по Δv, м/с' % (
        what, len(STAND_OFFSETS), STAND_OFFSETS[-1] * st.TICK))
    print('    %-16s' % '' + ''.join('%6s' % dv(m) for m in stand['mags']))
    for d in DIRS8:
        print('    %-16s' % DIR_LABELS[d] + ''.join('%6d' % n for n in stand['falls'][d]))
    _, falls, runs = judged(stand)
    print('    до %s м/с: %d из %d (можно до %d)%s' % (dv(JUDGED_MAG), falls, runs, STAND_FALLS,
                                                    '; до толчка %d' % stand['early'] if stand['early'] else ''))
    by_mag = [sum(walk['falls'][d][k] for d in DIRS8) for k in range(len(walk['mags']))]
    where = walk_where(walk)
    print('  толчки на ходу 0.5 м/с (%s, %d фаз шага): %d из %d (Δv %s: %s)%s%s' % (
        what, len(WALK_OFFSETS), sum(by_mag), sum(sum(n) for n in walk['runs'].values()),
        ' / '.join(dv(m) for m in walk['mags']), ' / '.join('%d' % n for n in by_mag), '; ' + where if where else '',
        '; до толчка %d' % walk['early'] if walk['early'] else ''))


def print_pushes(summary):
    """The 30 s turn, the height checks (stages with the height command), the judged pushes of step 2 (with the foot
    capsules), the ±1 m/s side check and the pushes without the capsules (the sim2sim forecast)."""
    turn = summary['stand30']
    print('  стояние 30 с (Unitree +6 мс): ' + (
        'поворот %+.1f°' % turn['heading'] if turn['heading'] is not None
        else 'не измерено' if turn['fell'] is None else 'упал на %.1f с' % turn['fell'])
          + ('' if not summary.get('height') else ' (судится: |.| <= %g°, этап с командой высоты)' % STAND_TURN_LIMIT))
    if summary.get('height'):
        print_height(summary['height'], turn)
    print_push_tables(summary['pushes'], PUSH_PHYSICS, 'судится')
    side = summary['side1']
    print('  шаг вбок ±1 м/с (%s, судится по падениям): прогонов %d, падений %d%s; скорость вбок %s / %s м/с, увод '
          '%s / %s рад/с, стоп после шага %s м' % (
              PHYSICS_LABELS[PUSH_PHYSICS], side['runs'], len(side['falls']),
              ' (%s)' % '; '.join(side['falls']) if side['falls'] else '',
              *(number(side['speed'][key], '%+.2f') for key, _ in SIDE_CHECK_SEGMENTS),
              *(number(side['drift'][key], '%+.3f') for key, _ in SIDE_CHECK_SEGMENTS), number(side['stop'], '%.2f')))
    if summary.get('pushes_sim2sim'):
        print_push_tables(summary['pushes_sim2sim'], FORECAST_PHYSICS,
                          'прогноз sim2sim: стопы проходят друг сквозь друга, как в unitree_mujoco; не судится')


def print_height(height, turn):
    """The height command checks of one stage (height_check()), a row each: the crouched walk, the 30 s stand at 0.78
    (stand30()), the crouched stands; the judged limit or 'не судится' last."""
    print('  команда высоты (%s):' % variant_label(CROUCH_VARIANT))
    print('    %-26s %11s %10s %7s %7s  %s' % ('', 'увод, рад/с', 'поворот, °', 'v, м/с', 'таз, м', 'правило'))
    walk = height['walk']
    rows = [('ходьба 0.5 м/с на %.2f' % walk['height'], walk['drift'], None, walk['speed'], walk['pelvis'],
             '|увод| < %.2f' % CROUCH_DRIFT_LIMIT, walk['fell'], walk['runs'])]
    rows.append(('стояние 30 с на 0.78', None, turn['heading'], None, None, '|поворот| <= %g°' % STAND_TURN_LIMIT,
                 None if turn['fell'] is None else '%.1f с' % turn['fell'], 1))
    rows += [('стояние 30 с на %.2f' % s['height'], None, s['heading'], None, s['pelvis'], 'не судится', s['fell'],
              s['runs']) for s in height['stands']]
    for text, drift, heading, speed, pelvis, rule, fell, runs in rows:
        print('    %-26s %11s %10s %7s %7s  %s%s' % (
            text, number(drift, '%+.3f'), number(heading, '%+.1f'), number(speed, '%+.2f'), number(pelvis, '%.3f'),
            rule, '; упал: %s' % fell if fell else '' if runs else '; не прогнан'))


def print_height_table(stages):
    """After every stage: the stages with the height command, a row each with the height checks' numbers and step 2
    without them (the rule before) and with them."""
    rows = [s for s in stages if s['select'].get('height')]
    if not rows:
        return
    walk, stands = crouch_height(CROUCH_WALK, 'walk'), [crouch_height(plan, 'stand') for plan in CROUCH_STANDS]
    print('\nкоманда высоты (%s), этапы с height_command: увод на ходу 0.5 м/с на %.2f (|.| < %.2f), поворот стоя '
          '30 с на 0.78 (|.| <= %g°), на %s (не судятся)' % (
              variant_label(CROUCH_VARIANT), walk, CROUCH_DRIFT_LIMIT, STAND_TURN_LIMIT,
              ' / '.join('%.2f' % c for c in stands)))
    width = max(len(s['label']) for s in rows)
    print('  %-*s %9s %9s %9s %9s %14s %12s' % (width, '', 'увод %.2f' % walk, 'стоя 0.78',
                                                *('стоя %.2f' % c for c in stands), 'шаг 2 без них', 'шаг 2'))
    for s in rows:
        select, height = s['select'], s['select']['height']
        before = len(select['reasons']) == len(height['reasons'])  # every other reason is the rule before's
        print('  %-*s %9s %9s %9s %9s %14s %12s' % (
            width, s['label'], 'упал' if height['walk']['fell'] else number(height['walk']['drift'], '%+.3f'),
            number(select['stand30']['heading'], '%+.1f'),
            *('упал' if x['fell'] else number(x['heading'], '%+.1f') for x in height['stands']),
            'проходит' if before else 'не проходит', 'проходит' if select['passes'] else 'не проходит'))


def verdict(summary):
    """The verdict and the reasons; those of the height checks (the last ones) after the others, apart."""
    if summary['passes']:
        print('  итог: проходит')
        return
    height = (summary.get('height') or {}).get('reasons') or []
    rest = summary['reasons'][:len(summary['reasons']) - len(height)]
    print('  итог: не проходит: ' + ' | '.join((['; '.join(rest)] if rest else [])
                                                + (['команда высоты: ' + '; '.join(height)] if height else [])))


def report(stage):
    print('\n== %s' % stage['label'])
    print_rows('шаг 2 — 7 вариантов физики и толчки', stage['select'], [(n, t) for n, t, _ in VARIANTS], side05=True)
    print_pushes(stage['select'])
    if stage.get('feet'):
        print('  ' + feet.line(stage['feet']))
    if stage['select'].get('chatter'):
        print('  ' + ch.line(stage['select']['chatter']))
    verdict(stage['select'])
    if stage.get('fresh'):
        print_rows('шаг 3 — свежая проверка (Unitree +6 мс)', stage['fresh'], [(n, t) for n, t, _ in FRESH])
        verdict(stage['fresh'])
    print('  время: шаг 2 — %d прогонов, %.0f с процессов%s' % (
        stage['runs']['select'], stage['seconds']['select'],
        '' if not stage.get('fresh') else '; шаг 3 — %d прогонов, %.0f с процессов' % (
            stage['runs']['fresh'], stage['seconds']['fresh'])))


def cost(kind, job):
    """Rough run length in physics ticks, to start the long runs first."""
    if kind == 'feet':
        return feet.cost(job)
    if kind == 'chatter':  # no FixStand, our reset
        return job['seconds'] / st.TICK + 700
    if kind == 'push':
        plan, physics = st.PLANS['kstand' if job['plan'] == 'stand' else 'kwalk'], job['phys']
    else:
        plan, physics = st.PLANS[job['plan']], job['physics']
    fix = 3500 if physics.startswith('um') else 14000  # FixStand at 2 ms / 0.5 ms
    return sum(segment[4] for segment in plan) / st.TICK + fix + 700  # (name, vx, vy, wz, seconds[, height])


def init():
    import torch
    torch.set_num_threads(1)


def work(item):
    index, kind, job = item
    t0 = time.perf_counter()
    if kind == 'push':
        um.Stepper, kick.PLAN = kick.PushStepper, job['plan']
        try:
            result = kick.run(job)
        finally:
            um.Stepper = STEPPER
    elif kind == 'feet':  # sets and restores its own stepper and push plan
        result = feet.run(job)
    elif kind == 'chatter':  # sets and restores its own recording of the motor torque
        result = ch.run(job)
    else:
        result = st.run(job)
    return index, result, time.perf_counter() - t0


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Отбор этапа: 7 вариантов физики и толчки в копии связки (шаг 2), '
                                                 'свежая проверка (шаг 3)')
    parser.add_argument('stages', nargs='+', type=Path, help='папка этапа или выгрузки export_g1')
    parser.add_argument('--fresh', action='store_true', help='ещё шаг 3: свежая проверка')
    parser.add_argument('--workers', type=int, default=8, help='рабочих процессов (пока идёт обучение, не больше 8)')
    parser.add_argument('--out', type=Path, help='JSON: итоги и все прогоны')
    args = parser.parse_args(argv)
    paths = [p.resolve() for p in args.stages]
    for p in paths:
        if not ((p / 'policy.onnx').is_file() or ((p / 'metadata.json').is_file() and (p / 'policy.zip').is_file())):
            raise SystemExit(f'Не этап и не выгрузка export_g1: {p}')
    if not st.LOGS.is_dir():
        raise SystemExit(f'Нет логов связки (окна времени): {st.LOGS} — запущена ли WSL Ubuntu?')
    heights = [st.has_height(p) for p in paths]  # task_config height_command: the height checks of step 2
    arms = [ch.has_arm_targets(str(p)) for p in paths]  # task_config arm_targets: chatter's carry scenario
    delays = [ch.stage_delay(str(p)) for p in paths]  # the training delay, which chatter's reference runs at
    items = []  # (stage index, step, kind, job); the chatter reference's: (None, 'reference', 'chatter', job)
    for i, p in enumerate(paths):
        items += [(i, 'select', kind, job) for kind, job in select_jobs(p, heights[i], arms[i])]
        if args.fresh:
            items += [(i, 'fresh', kind, job) for kind, job in fresh_jobs(p)]
    for delay in sorted(set(delays)):
        items += [(None, 'reference', 'chatter', job) for job in ch.reference_jobs(delay)]
    order = sorted(range(len(items)), key=lambda k: -cost(items[k][2], items[k][3]))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):  # one core per worker
        os.environ.setdefault(key, '1')
    print('этапов: %d, прогонов: %d, процессов: %d' % (len(paths), len(items), args.workers), flush=True)
    results, seconds = [None] * len(items), [0.] * len(items)
    t0 = time.time()
    with Pool(args.workers, initializer=init, maxtasksperchild=16) as pool:
        done = 0
        for index, result, secs in pool.imap_unordered(work, [(k, items[k][2], items[k][3]) for k in order]):
            results[index], seconds[index] = result, secs
            done += 1
            if done % 100 == 0 or done == len(items):
                print('  %d из %d, %.0f с' % (done, len(items), time.time() - t0), flush=True)
    wall = time.time() - t0
    reference = {delay: [results[k] for k, item in enumerate(items) if item[1] == 'reference'
                         and item[3]['delay'] == delay] for delay in set(delays)}
    stages = []
    for i, p in enumerate(paths):
        def of(step, kind):
            return [results[k] for k, item in enumerate(items) if item[:3] == (i, step, kind)]

        def spent(step):
            return sum(seconds[k] for k, item in enumerate(items) if item[:2] == (i, step))

        chatter = ch.summarize(of('select', 'chatter'), reference[delays[i]])
        stage = dict(label=label(p), path=str(p), height_command=heights[i],
                     select=select_summary(of('select', 'run'), of('select', 'push'), heights[i], chatter),
                     feet=feet.summary(of('select', 'feet')),
                     runs={'select': sum(item[:2] == (i, 'select') for item in items)},
                     seconds={'select': spent('select')})
        if args.fresh:
            stage['fresh'] = fresh_summary(of('fresh', 'run'))
            stage['runs']['fresh'] = sum(item[:2] == (i, 'fresh') for item in items)
            stage['seconds']['fresh'] = spent('fresh')
        report(stage)
        stage['results'] = {step: {kind: of(step, kind) for kind in ('run', 'push', 'feet', 'chatter')}
                            for step in (('select', 'fresh') if args.fresh else ('select',))}
        stages.append(stage)
    print_height_table(stages)
    passed = [s for s in stages if s['select']['passes']]
    print('\nшаг 2 проходят %d из %d этапов: %s' % (len(passed), len(stages), ', '.join(s['label'] for s in passed)
                                                    or '-'))
    if args.fresh:
        both = [s for s in passed if s['fresh']['passes']]
        print('шаг 2 и шаг 3 проходят %d из %d: %s' % (len(both), len(stages), ', '.join(s['label'] for s in both)
                                                       or '-'))
    print('%.0f с на %d процессах (%.0f с на этап)' % (wall, args.workers, wall / len(stages)))
    if args.out:
        rule = dict(drift=DRIFT_LIMIT, stop=STOP_LIMIT, variants=[name for name, _, _ in VARIANTS], side05=SIDE05,
                    push_physics=PUSH_PHYSICS, forecast_physics=FORECAST_PHYSICS, side_check=SIDE_CHECK,
                    height=dict(walk=CROUCH_WALK, stands=CROUCH_STANDS, variant=CROUCH_VARIANT,
                                drift=CROUCH_DRIFT_LIMIT, stand_turn=STAND_TURN_LIMIT),
                    pushes=dict(dirs=DIRS8, stand_mags=STAND_MAGS, stand_offsets=STAND_OFFSETS, walk_mags=WALK_MAGS,
                                walk_offsets=WALK_OFFSETS, judged_mag=JUDGED_MAG, stand_falls=STAND_FALLS,
                                cell_falls=CELL_FALLS, half=HALF, walk_falls=WALK_FALLS),
                    chatter=dict(reference=ch.REFERENCE, scenarios=ch.BASE, seeds=ch.SEEDS, seconds=ch.SECONDS,
                                 gated=ch.GATED, factor=ch.GATE_FACTOR, gate=ch.gate_text()))
        args.out.write_text(json.dumps(dict(rule=rule, workers=args.workers, seconds=wall, stages=stages,
                                            chatter_reference={'%g' % d: runs for d, runs in reference.items()}),
                                       indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
