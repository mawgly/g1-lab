"""Bounded task parameters; callers never supply arbitrary environment kwargs."""
import math
import re

# g1_velocity v2 weights. A key missing from a saved config takes this value, so terms added later
# default to "absent" (forward, sideways, torso, orientation, height = 0) and old checkpoints keep their reward.
G1_REWARD_WEIGHTS = dict(velocity=2., forward=0., sideways=0., heading=0., yaw=1., yaw_drift=0., alive=.5, gait=.5,
                         swing=10., tilt=1.,
                         orientation=0., height=0., height_ceiling=0., torso=0., base_motion=2., still=0.,
                         stand_pose=0., slip=0., feet_flat=0., clearance=0., self_contact=0., arm_press=0., arm_track=0.,
                         arm_error=0., posture=.1, hips=0., effort=1e-5, action_change=.01, fall=5.)
# Values the browser form starts with for a new experiment.
G1_RECOMMENDED = {**G1_REWARD_WEIGHTS, 'torso': 2.}
# g1_getup weights (iteration 33, the critic's specification; tasks/g1_tensor_math.getup_reward_components): the task
# (head height 1, torso upright 0.5), the standing state (bonus 1, pose 0.5, pelvis stillness), regularization always
# (torques 2e-5, joint speeds 1e-4, action change as the walking task's 0.01, legs beyond the soft joint limits).
# stand_still and joint_limits were not given numbers: 1 each, to be tuned by branches. The action change is the walking
# task's default, not candidate 8's 0.05: the gate requires that standing in home under σ 1 action noise earns more than
# lying still (tests/test_g1_getup_gate.py), and at 0.05 the noise alone costs 2.39 per step (standing −0.15 against
# the best lying 0.46; 0.84 at σ 0.8), at 0.01 0.48 (standing 1.76).
# impact (iteration 35, working value 2): the speed at which a body other than the feet hits the floor
# (g1_tensor_math.impact_penalty); 0, the default, is the original task bit for bit and computes nothing. A reward
# weight only: the contract and branching are unchanged.
# sole_impact (iteration 44, working value 2): the same charge on the sole spheres, which impact leaves out, above 0.5
# m/s (g1_tensor_math.GETUP_SOLE_IMPACT_SPEED; landing a jump on the feet); 0, the default, is the task before it bit
# for bit and computes nothing. A reward weight only, as impact.
# stance_width (after branch 46, working value 2): the feet's width beyond 0.30 m while the pelvis is above 0.5 m and
# both soles are down (g1_tensor_math.GETUP_STANCE_WIDTH, stance_active); 0, the default, is the task before it bit for
# bit and computes nothing. A reward weight only.
# arm_press (iteration 52, working value 0.05): standing only (getup_standing), Σ over the free arm joints (the locked
# wrists left out) of (PD target before pd_bound − angle after the step)², as walking's arm_press; 0, the default, is
# the task before it bit for bit and computes nothing. A reward weight only.
# style (iteration 53, working value 0.5): +w·exp(−d/2) while the head top is 0.3-1.1 m up, d the distance to the
# nearest human frame of the style library (task_config style_library; g1_tensor_math.style_distance); 0, the default,
# is the task before it bit for bit: nothing is loaded or computed. A reward weight only.
# head_rise (iteration 57, working value 3): −w·max(0, v_z − 0.8 m/s), v_z the head top's vertical speed over the
# control step, every step (g1_tensor_math.head_rise); 0, the default, is the task before it bit for bit and computes
# nothing. A reward weight only.
# sole_slip (iteration 62, working value 0.1): −w·Σ over the feet with the sole below 1 cm of their horizontal speed²
# (g1_tensor_math.sole_slip); 0, the default, is the task before it bit for bit and computes nothing. A reward weight
# only.
# amp (iteration 55, working value 1): +w·[0.3 < head < 1.1]·max(0, 1 − 0.25·(D − 1)²), D the AMP discriminator of
# the step's state pair (humanoid_sim/getup_amp), paid by the full-GPU learner (TensorPPO.collect), not by the task: the
# CPU task and its stage evaluation do not see it (no reward component). 0, the default, builds no discriminator.
G1_GETUP_REWARD_WEIGHTS = dict(head=1., upright=.5, stand=1., stand_pose=.5, stand_still=1., effort=2e-5,
                               joint_speed=1e-4, action_change=.01, joint_limits=1., impact=0., sole_impact=0.,
                               stance_width=0., arm_press=0., style=0., head_rise=0., sole_slip=0., amp=0.)
# The reward weights the task itself pays (its reward components); amp is paid by the learner.
G1_GETUP_TASK_WEIGHTS = tuple(key for key in G1_GETUP_REWARD_WEIGHTS if key != 'amp')
# g1_getup's robot, fixed by the task (not settings; part of its identity): the walking candidates' real-robot model
# with Unitree's foot and hand capsules, locked wrists, the knee floor and the action box, and the walking observation
# layout (history 5, the phase stopped, critic extras) with heading hold off; the command is always zero.
G1_GETUP_PROFILE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, lock_wrists=True,
                        knee_floor=.061, foot_collision=True, hand_collision=True, action_limit=10.,
                        heading_hold=False, stand_heading_hold=False, side_heading_hold=False,
                        gait=dict(period=.6, stance_fraction=.55))
G1_LABELS = dict(velocity='Следование скорости', forward='Скорость вперёд: чем быстрее, тем больше (до 4 м/с)',
                 sideways='Штраф за бег боком (боковая скорость таза)',
                 heading='Штраф за корпус, отвёрнутый от курса (бег наискось)', yaw='Следование повороту / курсу',
                 yaw_drift=('Штраф за увод при команде «прямо»: |среднее за ~1,2 с (поворот таза − команда)|, '
                            'линейный'),
                 alive='Сохранение равновесия',
                 gait='Ритм шагов (контакт стоп)', swing='Штраф за высоту маха', tilt='Штраф за наклон таза (1 − cos)',
                 orientation='Штраф за наклон таза, квадратичный (как у Unitree)',
                 height='Штраф за высоту таза (цель 0,78 м)',
                 height_ceiling=('Штраф за таз выше команды приседа (при обычной высоте 0,78 м — ноль; нужна команда '
                                 'высоты)'),
                 torso='Штраф за наклон корпуса (цель — 6° вперёд)', base_motion='Штраф за раскачивание корпуса',
                 still='Штраф за дрейф и разворот на месте (при команде «стоять»)',
                 stand_pose='Штраф за отход ног от исходной позы стоя (при команде «стоять»)',
                 slip='Штраф за скольжение стоп по земле (как feet_slide у Unitree)',
                 feet_flat='Штраф за наклон стоп (стопа на пятке или на носке), квадратичный',
                 clearance=('Штраф за низкий перенос стопы при ходьбе: |подошва − 0,10 м| × скорость стопы '
                            '(как feet_clearance у Unitree)'),
                 self_contact=('Штраф за касание тела с собой сильнее 10 Н (руки упираются в корпус или бёдра): '
                               'доля под-шагов физики, как self_collisions у Unitree'),
                 arm_press=('Штраф за давление рук (упор в тело): сумма по суставам рук (цель мотора − угол)², '
                            'рад²; плавно падает, когда упор слабеет'),
                 arm_track=('Руки по целям: среднее по 10 свободным суставам рук exp(−ошибка²/0,3²) (только при '
                            'командах рук)'),
                 arm_error=('Штраф за ошибку рук на удержании цели: среднее |ошибка| по 10 свободным суставам рук, '
                            'рад, линейный (только при командах рук, не во время перехода)'),
                 posture='Штраф за отклонение рук, корпуса и бёдер от исходной позы',
                 hips='Штраф за разведение и разворот бёдер (ноги под корпусом)', effort='Штраф за усилие моторов',
                 action_change='Штраф за резкие действия', fall='Штраф за падение')
# Training command range (forward speed, m/s, negative = backwards; sideways and turning ranges are
# symmetric) and the share of "stand" commands. Missing keys mean the original forward-only range, so
# checkpoints saved before these settings keep their task.
# straight_probability: share of walking commands without a turn when turns are drawn (yaw_rate > 0); with
# heading hold the robot keeps the heading it had when such a command came (tasks/g1_tensor_math.sample_commands).
# side_share: share of walking commands that go purely sideways (vx 0, no turn), taken from the backward ones, so at
# most their share −min_speed / (max_speed − min_speed); side_stop: after such a command, the probability that a
# fresh command other than forward becomes "stand" (iteration 19: in the Unitree bridge copy the stages fell only
# on stops after a sideways step). Both 0 (missing) is the original draw, bit for bit.
# side_speed (needs side_share): the top sideways speed of the purely sideways commands alone (the others keep
# ±lateral_speed; iteration 25: side steps to ±1 m/s with the diagonals and forward commands unchanged); 0 or null
# (missing) is off, the original draw bit for bit, otherwise above SIDE_MIN_SPEED.
# crouch_share, crouch_min (pairs: standing, walking) and crouch_speed (task_config height_command, iteration 28): the
# share of training commands that come with a crouch, the lowest pelvis height drawn for them (m, uniform up to
# CROUCH_HEIGHTS[1]) and the top speed of a walking crouch (g1_tensor_math.height_draw, at every reset and command
# change). A share needs height_command; [0, 0] (missing) draws no crouch.
CROUCH_HEIGHTS = (.6, .78)  # g1_tensor_math.CROUCH_TABLE's range
CROUCH_PAIRS = ('crouch_share', 'crouch_min')
G1_COMMANDS = dict(min_speed=0., max_speed=1.5, stand_probability=.2, lateral_speed=0., yaw_rate=0.,
                   straight_probability=0., side_share=0., side_stop=0., side_speed=0., crouch_share=[0., 0.],
                   crouch_min=[.64, .70], crouch_speed=1.)
G1_COMMAND_LIMITS = dict(min_speed=(-1., 2.), max_speed=(0., 2.), stand_probability=(0., 1.),
                         lateral_speed=(0., 1.), yaw_rate=(0., 2.), straight_probability=(0., 1.),
                         side_share=(0., 1.), side_stop=(0., 1.), side_speed=(0., 1.5), crouch_share=(0., 1.),
                         crouch_min=CROUCH_HEIGHTS, crouch_speed=(0., 2.))
G1_COMMAND_LABELS = dict(min_speed='Скорость команд от, м/с (минус — назад)', max_speed='Скорость команд до, м/с',
                         stand_probability='Доля команд «стоять» (0–1)', lateral_speed='Команды вбок до ±, м/с',
                         yaw_rate='Команды поворота до ±, рад/с',
                         straight_probability='Доля команд «прямо» без поворота, курс держится (0–1)',
                         side_share='Доля команд «чисто вбок» без поворота, за счёт команд назад (0–1)',
                         side_stop=('Стоп после шага вбок: вероятность, что следующая команда (кроме «вперёд») — '
                                    '«стоять» (0–1)'),
                         side_speed=('Команды «чисто вбок» до ±, м/с (0 — как у остальных команд вбок; диагонали '
                                     'и «вперёд» не меняются)'),
                         crouch_share=('Доля команд с приседом: стоя и на ходу, пара чисел 0–1 (нужна команда '
                                       'высоты)'),
                         crouch_min='Нижняя высота таза в приседе, м: стоя и на ходу, пара чисел 0,60–0,78',
                         crouch_speed='Присед на ходу — только до этой скорости, м/с')
# Lowest sideways speed of a purely sideways command: g1_tensor_math.STAND_SPEED (below it the task asks to stand).
SIDE_MIN_SPEED = .1
# Gait clock rewarded by the `gait` term: cycle length (s) and the share of it each foot is on the ground.
# A share below 0.5 leaves flight phases, as in running. Missing keys mean the original walking clock.
G1_GAIT = dict(period=.8, stance_fraction=.55)
G1_GAIT_LIMITS = dict(period=(.4, 1.2), stance_fraction=(.2, .8))
G1_GAIT_LABELS = dict(period='Цикл шага, с', stance_fraction='Доля цикла на опоре (меньше 0,5 — бег с полётом)')
# Policy actions are clipped to ±action_limit before target = home + scale·action. 1 is the original
# contract (knee target within ±20° of the default pose); a wider limit opens more of the real joint
# range (joint limits and motor torques still apply) and is a different action contract.
ACTION_LIMIT_RANGE = (1., 10.)
# real_robot: Unitree's deployable setup (unitree_rl_mjlab G1 velocity v0) — rotor armature and motor
# torque-speed curves with friction, observations exactly as the robot's controller builds them (no base
# linear velocity, 0.6 s gait phase), sensor noise and physics randomisation in training. See
# docs/g1-real-robot-plan.md. A different model and observation contract.
DEPLOY_PHASE_PERIOD = .6
# critic_extras (real robot only): the observation ends with base linear velocity and pelvis height, exact from
# the simulator, for the PPO critic alone; the actor never sees them (policies.CriticExtrasPolicy), so the
# deployed policy still uses only what the robot measures. Asymmetric actor-critic as in legged_gym and mjlab.
CRITIC_EXTRAS = 4
# g1_getup's critic extras: those four, then the head height and the assist force (tasks/g1_tensor_math.
# getup_critic_extras).
GETUP_CRITIC_EXTRAS = 6
# history (real robot only): frames of observation history per term, as Unitree's controller builds them with
# history_length (unitree_rl_lab G1 velocity uses 5): each term's frames oldest first, terms in deploy.yaml order.
# A different observation contract (obs-unitree-deploy-v0-hN).
HISTORY_RANGE = (1, 10)
# training_realism (real robot, CUDA training only): scales sensor noise, encoder bias, randomisation ranges and
# pushes from none (0) to Unitree's full setup (1), for a curriculum. Physics, observations and evaluation are
# unaffected, so it is not part of the policy contract; final policies are trained and judged at 1.
# action_delay: the largest delay (s) from policy output to motor target in training. Each training episode draws
# k uniformly from 0..round(action_delay / physics step) substeps (tasks/g1_tensor_math.action_lags); resets with
# a command (evaluation, scenarios) use k = 0 unless reset options give action_delay_steps. Observations and
# actions are unchanged, so it is not part of the policy contract. 0 (missing) is the original task.
ACTION_DELAY_RANGE = (0., .05)
# arm_perturb: the share of training time during which the arm joints follow a held "perturbation" action instead of
# the policy's (segments of 0.5-2 s per world; tasks/g1_tensor_math.arm_perturb_draw). The applied action replaces the
# policy's everywhere after it (PD, delay queue, reward, observed previous action); resets with a command and
# set_command (evaluation, scenarios) have none. Observations and actions are unchanged, so it is not part of the
# policy contract. 0 (missing) is the original task.
ARM_PERTURB_RANGE = (0., .5)
# arm_perturb_radians (iteration 30): the amplitude of a segment's random arm pose, ±rad per joint around home
# (tasks/g1_tensor_math.arm_perturb_draw; the box and the joint ranges still clip it). ARM_PERTURB_RADIANS (missing)
# is the original draw, bit for bit; another value needs arm_perturb. arm_perturb_contact_mask (iteration 30): while a
# world's arms follow a segment, contacts touching an arm link do not count in the self_contact share
# (tasks/g1_tensor_math.self_contact_pairs): the policy is not charged for a pose it did not choose; contacts of the
# legs, feet, torso and pelvis count as before. Needs arm_perturb; false (missing) is the original term. Neither
# changes observations or actions, so neither is part of the policy contract.
ARM_PERTURB_RADIANS = .4
ARM_PERTURB_RADIANS_RANGE = (.1, 1.6)
# force_push: the largest Δv (m/s) of the training pushes as a force on the pelvis, M·Δv/T for T 0.05-0.2 s, every
# 2-4 s per world, the direction uniform (tasks/g1_tensor_math.force_push_draw; the sim2sim push patch's force; 1.25 is
# the working value of iteration 20). On top of the CUDA task's velocity pushes, on both backends, not scaled by
# training_realism; resets with a command and set_command (evaluation, scenarios) have none. Observations and actions
# are unchanged, so it is not part of the policy contract. 0 (missing) is the original task.
FORCE_PUSH_RANGE = (0., 2.)
# motor_friction (real robot only): the largest multiplier of the motor model's joint friction (static and viscous,
# tasks/g1_tensor_math.motor_torque) in training. Each training episode draws m ~ U(0, motor_friction) per world
# (motor_friction_draw; 1.25 is the working value of iteration 27); resets with a command (evaluation, scenarios) keep
# m = 1 unless reset options give motor_friction_scale. Not scaled by training_realism. Observations and actions are
# unchanged, so it is not part of the policy contract. 0 (missing or null) is the original task.
MOTOR_FRICTION_RANGE = (0., 2.)
# lock_wrists (29-DoF G1): the wrist pitch and yaw joints always get action 0, so their motors hold the home pose
# (tasks/g1_tensor_math.lock_joints); the applied action is the policy's with those entries zeroed, in everything
# after it. Same layout, different meaning of four action entries: its own action contract (action_schema
# …-wristlock). The legs-only G1 has no wrists and rejects it. False (missing) is the original task.
# knee_floor (rad, both G1 models): the knee PD target is max(target, knee_floor) (tasks/g1_tensor_math.target_low),
# the observed previous action stays the policy's, as g1_ctrl's last_action; 0.061 is Unitree's soft limit (0.9 of
# the range). Same layout, different meaning of the knee actions below the floor: its own action contract
# (action_schema …-knee<floor>). Missing or null is the original task (the knee target down to the joint range).
KNEE_FLOOR_RANGE = (0., 1.)
# height_command (real robot only, iteration 28): the pelvis height c the policy is told to keep, one observation term
# [c − 0.78] after last_action, drawn at every reset and command change (commands crouch_share / crouch_min /
# crouch_speed); the fall height, the height term, height_ceiling and the stand_pose leg pose follow it
# (tasks/g1_tensor_math.height_draw, stand_target, fall_height). upper_body_slot (real robot only): a reserved term of
# 18 zeros after it (a flag and 17 waist and arm targets) for a later upper-body command. Each is its own observation
# contract (observation_schema …-hcmd, …-ubslot). False (missing) is the original layout, bit for bit.
# foot_collision (both G1 models): the feet collide with each other through Unitree's foot capsules (unitree_rl_mjlab
# g1.xml: 7 per foot, radius 0.01; tasks/g1_velocity.FOOT_CAPSULES, with Unitree's contact: condim 3, friction 0.6),
# which touch nothing else, so the floor contact stays the sole spheres; a foot-foot contact above 10 N is charged by
# the self_contact term. Ours alone passed through
# each other (iteration 23). Observations and actions are unchanged, the physics is not: its own model revision
# (FOOT_COLLISION_SUFFIX), a stage moves there with warm_start.carry_over. False (missing) is the original model.
FOOT_COLLISION_SUFFIX = '-footcol'
# hand_collision (29-DoF G1, iteration 33): Unitree's hand capsules (unitree_rl_mjlab g1.xml left/right_hand_collision:
# radius 0.035 on the wrist_yaw links; tasks/g1_velocity.HAND_CAPSULES) collide with the floor and the robot; ours had
# no hand collision at all, so leaning on a hand rested on the wrist edge. Observations and actions are unchanged, the
# physics is not: its own model revision (HAND_COLLISION_SUFFIX, after FOOT_COLLISION_SUFFIX), a stage moves there with
# warm_start.carry_over. False (missing) is the original model.
HAND_COLLISION_SUFFIX = '-handcol'
# arm_blind (real robot only, iteration 31): the policy does not see its arms. The observation is unchanged; its arm
# columns (joint angles, velocities and last actions of the shoulders, elbows and wrists, every history frame) keep a
# fixed normalization that maps them onto one constant, the parent stage's own mean normalized arm inputs
# (tasks/g1_tensor_math.ARM_BLIND_SCALE, G1VelocityEnv.fixed_normalization; written by warm_start.blind_arms,
# scripts/lab/blind_branch.py), frozen in training. Same layout, different content: its own observation contract
# (observation_schema …-armblind, ARM_BLIND_SUFFIX). False (missing) is the original task, bit for bit.
ARM_BLIND_SUFFIX = '-armblind'
# arm_targets (real robot, 29-DoF, needs upper_body_slot and lock_wrists; iteration 40, stage 3 "arms to targets"): the
# upper-body slot carries an arm command, flag 1 and the arm joints' targets relative to home, and the arm action is a
# residual on it (target = home + d + scale·a; the observed last action clip(a) + d/scale). Each training episode draws
# free arms (flag 0: the old task), arms at home or random targets with hand loads on a curve; resets with a command
# keep flag 0 (tasks/g1_tensor_math ARM_MODES ... load_curve_update). The policy sees its arms (not with arm_blind): its
# own observation contract (…-armtgt, ARM_TARGETS_SUFFIX) and action contract (…-armres, ARM_RESIDUAL_SUFFIX); a stage
# moves there with warm_start.unblind_arms (scripts/lab/arm_branch.py). False (missing) is the original task, bit for
# bit.
ARM_TARGETS_SUFFIX = '-armtgt'
ARM_RESIDUAL_SUFFIX = '-armres'
# load_curve (needs arm_targets; iteration 40b): the hand load ceiling decides once every `block` rollout windows of the
# full-GPU trainer on their summed flag-1 counts, up below raise_falls, down above lower_falls (falls per episode of
# max_episode_steps; tasks/g1_tensor_math.LoadCurveBlock); the step, the 2 kg ceiling and the arm_track threshold stay.
# Training only: the CPU task, the contract and branching are unchanged. None (missing) is the per-window rule.
LOAD_CURVE_KEYS = ('block', 'raise_falls', 'lower_falls')
LOAD_CURVE_BLOCK_RANGE = (1, 1000)


def load_curve_setting(value):
    """task_config load_curve normalized: None, or {'block': whole number in LOAD_CURVE_BLOCK_RANGE, 'raise_falls',
    'lower_falls': shares in [0, 1], raise_falls ≤ lower_falls}; anything else raises ValueError."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(LOAD_CURVE_KEYS):
        raise ValueError('G1 load_curve must be null or {"block": n, "raise_falls": share, "lower_falls": share}')
    block, low, high = value['block'], value['raise_falls'], value['lower_falls']
    if type(block) not in (int, float) or block != int(block) or not LOAD_CURVE_BLOCK_RANGE[0] <= block <= \
            LOAD_CURVE_BLOCK_RANGE[1]:
        raise ValueError(f'G1 load_curve block must be a whole number in [{LOAD_CURVE_BLOCK_RANGE[0]},'
                         f'{LOAD_CURVE_BLOCK_RANGE[1]}] rollouts')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in (low, high)) or low > high:
        raise ValueError('G1 load_curve raise_falls and lower_falls must be shares in [0,1], raise_falls <= lower_falls')
    return {'block': int(block), 'raise_falls': float(low), 'lower_falls': float(high)}
G1_PARAMETERS = (
    dict(group=None, key='real_robot', label='Реальный робот: моторы, датчики и разброс Unitree', default=False,
         type='bool'),
    dict(group=None, key='critic_extras', label='Подсказки критику при обучении (скорость и высота таза)',
         default=False, type='bool'),
    dict(group=None, key='history', label='История наблюдений, кадров (как у Unitree; 1 — без истории)',
         default=1, min=HISTORY_RANGE[0], max=HISTORY_RANGE[1]),
    dict(group=None, key='training_realism', label='Реалистичность обучения: шум, разброс и толчки (0–1)',
         default=1., min=0., max=1.),
    dict(group=None, key='heading_hold', label='Держать курс (вперёд по оси X)', default=True, type='bool'),
    # Missing in a saved config means on (the original task); new experiments start with it off.
    dict(group=None, key='stand_heading_hold', default=False, type='bool',
         label='Держать курс и стоя (поворот на месте к курсу; выкл. — команда «стоять» нулевая, как в Isaac Lab)'),
    # Missing in a saved config means on (heading hold steered side steps too); new experiments start with it off.
    dict(group=None, key='side_heading_hold', default=False, type='bool',
         label=('Держать курс и при шаге «чисто вбок» (выкл. — поворот в команде нулевой, как от стика на роботе; '
                'увод виден награде)')),
    # Missing in a saved config means off (the phase always ticked); new real-robot runs match g1_ctrl.
    dict(group=None, key='phase_stop', default=True, type='bool',
         label='Фаза шага = 0, когда команда почти нулевая (как в программе Unitree на роботе)'),
    # Reward only (gait, swing, slip): a foot is on the ground when its lowest sole point is below
    # SOLE_CONTACT_HEIGHT, not when the ankle_roll origin is below CONTACT_HEIGHT. Missing means off.
    dict(group=None, key='sole_contact', default=False, type='bool',
         label='Опора стопы для награды — по нижней точке подошвы (а не по высоте голеностопа)'),
    # Reward only: the feet_flat term counts the feet in contact (the contact of gait and slip), not a foot in swing.
    # Missing means off (both feet, every step).
    dict(group=None, key='feet_flat_stance', default=False, type='bool',
         label='Штраф за наклон стоп — только для опорной стопы (перекат стопы в переносе не штрафуется)'),
    # Training episodes start at a random point of the gait cycle (either foot first); evaluation and set commands
    # keep phase 0. Missing means off (every episode started at phase 0, left foot first).
    dict(group=None, key='random_phase', default=True, type='bool',
         label='Случайная фаза шага в начале эпизода обучения (старт с любой ноги)'),
    dict(group=None, key='action_delay', label='Случайная задержка команды в обучении, с (0 — без задержки)',
         default=0., min=ACTION_DELAY_RANGE[0], max=ACTION_DELAY_RANGE[1]),
    dict(group=None, key='arm_perturb',
         label='Помехи на руки в обучении: доля времени, когда руки держат случайную позу (0 — без помех)',
         default=0., min=ARM_PERTURB_RANGE[0], max=ARM_PERTURB_RANGE[1]),
    # Missing in a saved config means ARM_PERTURB_RADIANS (the original draw); another value needs arm_perturb.
    dict(group=None, key='arm_perturb_radians',
         label='Помехи на руки: размах случайной позы каждого сустава руки, ± рад (0,4 — прежний)',
         default=ARM_PERTURB_RADIANS, min=ARM_PERTURB_RADIANS_RANGE[0], max=ARM_PERTURB_RADIANS_RANGE[1]),
    # Missing in a saved config means off (every robot-robot contact counted); needs arm_perturb.
    dict(group=None, key='arm_perturb_contact_mask', default=False, type='bool',
         label=('Во время помехи на руки касания рук не штрафуются (касания ног, стоп, корпуса и таза — '
                'как раньше)')),
    dict(group=None, key='force_push',
         label='Толчки силой на таз в обучении: наибольшее Δv, м/с (0 — без толчков; сила M·Δv/T раз в 2–4 с)',
         default=0., min=FORCE_PUSH_RANGE[0], max=FORCE_PUSH_RANGE[1]),
    # Real robot only; missing (or null) in a saved config means 0 (the motor friction as Unitree measured it).
    dict(group=None, key='motor_friction',
         label=('Разброс трения моторов в обучении: верхняя граница множителя трения суставов (0 — без разброса; '
                'на эпизод множитель от 0 до этого числа)'),
         default=0., min=MOTOR_FRICTION_RANGE[0], max=MOTOR_FRICTION_RANGE[1]),
    # Missing in a saved config means off (the policy drove the wrists); a different action contract.
    dict(group=None, key='lock_wrists', default=False, type='bool',
         label='Закрепить запястья (тангаж и рыскание) в исходной позе: команда на них всегда 0'),
    # Missing or null in a saved config means off (the knee target went down to the joint range); a different action
    # contract. The browser leaves the field empty for null.
    dict(group=None, key='knee_floor', default=None, min=KNEE_FLOOR_RANGE[0], max=KNEE_FLOOR_RANGE[1],
         label='Нижняя граница цели мотора колена, рад (пусто — без границы; 0,061 — мягкий предел Unitree)'),
    # Missing in a saved config means off (the feet passed through each other); a different model revision.
    dict(group=None, key='foot_collision', default=False, type='bool',
         label='Стопы сталкиваются друг с другом, как в модели Unitree (7 капсул на стопу; касание стоп идёт в штраф '
               'касания тела)'),
    # Missing in a saved config means off (the hands touched nothing); a different model revision; 29-DoF only.
    dict(group=None, key='hand_collision', default=False, type='bool',
         label='Кисти сталкиваются с полом и телом, как в модели Unitree (капсула на кисть)'),
    # Real robot only; missing in a saved config means off. Observation terms after last_action: their own observation
    # contract (…-hcmd, …-ubslot); a stage moves there with warm_start.insert_terms.
    dict(group=None, key='height_command', default=False, type='bool',
         label=('Команда высоты таза (присед): высота во входе политики, по ней падение, штраф высоты и поза стоя; '
                'при 0,78 м всё как раньше')),
    dict(group=None, key='upper_body_slot', default=False, type='bool',
         label=('Место во входе под команду верха тела: 18 нулей (флаг и цели талии и рук), пока не '
                'используется')),
    # Real robot only; missing in a saved config means off (the policy saw its arms). Its own observation contract
    # (…-armblind); a stage moves there with warm_start.blind_arms (scripts/lab/blind_branch.py).
    dict(group=None, key='arm_blind', default=False, type='bool',
         label=('Слепые руки: политика не видит углы, скорости и прошлые команды суставов рук (их входы — '
                'постоянные числа этапа-родителя)')),
    # Real robot only, needs upper_body_slot and lock_wrists; missing in a saved config means off (the slot stayed
    # zeros). Its own observation and action contracts (…-armtgt, …-armres); a stage moves there with
    # warm_start.unblind_arms (scripts/lab/arm_branch.py).
    dict(group=None, key='arm_targets', default=False, type='bool',
         label=('Руки по целям: в эпизодах обучения цели рук в слоте верха тела (флаг 1), действие рук — поправка к '
                'цели, груз на кистях; при команде сброса — свободные руки, как раньше')),
    # Needs arm_targets; missing or null in a saved config means the per-window rule (the first branch 40's).
    dict(group=None, key='load_curve', default=None,
         label=('Кривая груза на кистях: решение раз в block раскаток, вверх при доле падений ниже raise_falls, вниз '
                'выше lower_falls (пусто — решение каждую раскатку, 1% / 3%)')),
    dict(group=None, key='action_limit', label='Предел команды моторов (1 — прежний, 4 — почти весь ход суставов)',
         default=1., min=ACTION_LIMIT_RANGE[0], max=ACTION_LIMIT_RANGE[1]),
    *(dict(group='commands', key=key, label=G1_COMMAND_LABELS[key], default=G1_COMMANDS[key],
           min=G1_COMMAND_LIMITS[key][0], max=G1_COMMAND_LIMITS[key][1]) for key in G1_COMMANDS),
    *(dict(group='gait', key=key, label=G1_GAIT_LABELS[key], default=G1_GAIT[key],
           min=G1_GAIT_LIMITS[key][0], max=G1_GAIT_LIMITS[key][1]) for key in G1_GAIT),
    *(dict(group='reward_weights', key=key, label=G1_LABELS[key], default=G1_RECOMMENDED[key], min=0, max=100)
      for key in G1_REWARD_WEIGHTS),
)


def g1_reward_weights(values=None):
    values = {} if values is None else values
    if not isinstance(values, dict) or set(values)-set(G1_REWARD_WEIGHTS):
        raise ValueError('Unknown G1 reward weight')
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 100 for v in values.values()):
        raise ValueError('Reward weights must be finite and in [0,100]')
    return {**G1_REWARD_WEIGHTS, **values}


def _bounded(values, defaults, limits, what):
    values = {} if values is None else values
    if not isinstance(values, dict) or set(values)-set(defaults):
        raise ValueError(f'Unknown G1 {what} setting')
    result = {**defaults, **values}
    for key, (low, high) in limits.items():
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'G1 {what} {key} must be in [{low:g},{high:g}]')
    return {key: float(value) for key, value in result.items()}


def backward_share(min_speed, max_speed):
    """Share of walking commands drawn backwards (vx < 0) from the uniform speed range [min_speed, max_speed]."""
    if min_speed >= 0:
        return 0.
    return min(-min_speed / (max_speed - min_speed), 1.) if max_speed > min_speed else 1.


def _pair(value, limits, key):
    """A [standing, walking] pair of finite numbers within limits, as a list of floats."""
    low, high = limits
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
            type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high for v in value):
        raise ValueError(f'G1 command {key} must be two numbers (standing, walking) in [{low:g},{high:g}]')
    return [float(v) for v in value]


def g1_commands(values=None):
    if isinstance(values, dict) and values.get('side_speed', 0.) is None:
        values = {**values, 'side_speed': 0.}  # null: off, as missing
    pairs = {key: list(G1_COMMANDS[key]) for key in CROUCH_PAIRS}
    if isinstance(values, dict):  # the pairs are checked here, the numbers by _bounded
        pairs.update({key: _pair(values[key], G1_COMMAND_LIMITS[key], key) for key in CROUCH_PAIRS if key in values})
        values = {key: value for key, value in values.items() if key not in CROUCH_PAIRS}
    numbers = {key: value for key, value in G1_COMMANDS.items() if key not in CROUCH_PAIRS}
    result = {**_bounded(values, numbers, {key: G1_COMMAND_LIMITS[key] for key in numbers}, 'command'), **pairs}
    if result['min_speed'] > result['max_speed']:
        raise ValueError('G1 command min_speed exceeds max_speed')
    if result['side_share']:  # taken from the backward commands, with a real sideways step
        backward = backward_share(result['min_speed'], result['max_speed'])
        if result['side_share'] > backward:
            raise ValueError(f'G1 command side_share must not exceed the backward share {backward:g}')
        if result['lateral_speed'] <= SIDE_MIN_SPEED:
            raise ValueError(f'G1 command side_share needs lateral_speed above {SIDE_MIN_SPEED:g}')
    if result['side_stop'] and not result['side_share']:
        raise ValueError('G1 command side_stop requires side_share')
    if result['side_speed']:  # the purely sideways commands' own top speed, still a real sideways step
        if not result['side_share']:
            raise ValueError('G1 command side_speed requires side_share')
        if result['side_speed'] <= SIDE_MIN_SPEED:
            raise ValueError(f'G1 command side_speed must be 0 (off) or above {SIDE_MIN_SPEED:g}')
    return result


def g1_gait(values=None):
    return _bounded(values, G1_GAIT, G1_GAIT_LIMITS, 'gait')


def g1_config(values):
    """Always spell out every setting, so a checkpoint keeps its task if the defaults change later.

    heading_hold missing means off: checkpoints saved before it existed never had it. stand_heading_hold missing
    means on, for the same reason (heading hold used to steer while standing too). sole_contact missing means off
    (the reward judged contact by the ankle_roll origin height). random_phase missing means off (episodes started
    at phase 0). action_delay missing means 0 (no delay). arm_perturb missing means 0 (the policy drives the arms).
    lock_wrists missing means off (the policy drove the wrist pitch and yaw joints). knee_floor missing means None (no
    floor: the knee target went down to the joint range). force_push missing means 0 (no force pushes).
    feet_flat_stance missing means off (feet_flat charged both feet, every step). foot_collision missing means off (the
    feet passed through each other). side_heading_hold missing means on (heading hold steered purely sideways
    commands too). motor_friction missing or null means 0 (the motor model's friction as measured, every episode).
    height_command and upper_body_slot missing mean off (no such observation terms). arm_perturb_radians missing means
    ARM_PERTURB_RADIANS (the original arm poses), arm_perturb_contact_mask missing means off (arm contacts charged in a
    perturbation segment too). arm_blind missing means off (the policy saw its arms). hand_collision missing means off
    (the hands touched nothing). arm_targets missing means off (the upper-body slot stayed zeros). load_curve missing
    means None (the load curve decided every rollout window, 1 % / 3 %).
    """
    keys = {'reward_weights', 'heading_hold', 'commands', 'gait', 'action_limit', 'real_robot', 'critic_extras',
            'training_realism', 'history', 'stand_heading_hold', 'phase_stop', 'sole_contact', 'random_phase',
            'action_delay', 'arm_perturb', 'arm_perturb_radians', 'arm_perturb_contact_mask', 'lock_wrists',
            'knee_floor', 'force_push', 'feet_flat_stance', 'foot_collision', 'side_heading_hold', 'motor_friction',
            'height_command', 'upper_body_slot', 'arm_blind', 'hand_collision', 'arm_targets', 'load_curve'}
    if not isinstance(values, dict) or set(values)-keys:
        raise ValueError('Unknown G1 task setting')
    for key in ('heading_hold', 'real_robot', 'critic_extras', 'stand_heading_hold', 'phase_stop', 'sole_contact',
                'random_phase', 'arm_perturb_contact_mask', 'lock_wrists', 'feet_flat_stance', 'foot_collision',
                'side_heading_hold', 'height_command', 'upper_body_slot', 'arm_blind', 'hand_collision',
                'arm_targets'):
        if type(values.get(key, False)) is not bool:
            raise ValueError(f'{key} must be true or false')
    if values.get('critic_extras') and not values.get('real_robot'):
        raise ValueError('critic_extras requires real_robot')
    limit = values.get('action_limit', 1.)
    low, high = ACTION_LIMIT_RANGE
    if type(limit) not in (int, float) or not math.isfinite(limit) or not low <= limit <= high:
        raise ValueError(f'G1 action_limit must be in [{low:g},{high:g}]')
    real = values.get('real_robot', False)
    realism = values.get('training_realism', 1.)
    if type(realism) not in (int, float) or not math.isfinite(realism) or not 0 <= realism <= 1:
        raise ValueError('G1 training_realism must be in [0,1]')
    delay = values.get('action_delay', 0.)
    low, high = ACTION_DELAY_RANGE
    if type(delay) not in (int, float) or not math.isfinite(delay) or not low <= delay <= high:
        raise ValueError(f'G1 action_delay must be in [{low:g},{high:g}] s')
    perturb = values.get('arm_perturb', 0.)
    low, high = ARM_PERTURB_RANGE
    if type(perturb) not in (int, float) or not math.isfinite(perturb) or not low <= perturb <= high:
        raise ValueError(f'G1 arm_perturb must be in [{low:g},{high:g}]')
    radians = values.get('arm_perturb_radians', ARM_PERTURB_RADIANS)
    low, high = ARM_PERTURB_RADIANS_RANGE
    if type(radians) not in (int, float) or not math.isfinite(radians) or not low <= radians <= high:
        raise ValueError(f'G1 arm_perturb_radians must be in [{low:g},{high:g}] rad')
    if radians != ARM_PERTURB_RADIANS and not perturb:  # the amplitude of segments that never start
        raise ValueError('arm_perturb_radians requires arm_perturb')
    if values.get('arm_perturb_contact_mask') and not perturb:
        raise ValueError('arm_perturb_contact_mask requires arm_perturb')
    push = values.get('force_push', 0.)
    low, high = FORCE_PUSH_RANGE
    if type(push) not in (int, float) or not math.isfinite(push) or not low <= push <= high:
        raise ValueError(f'G1 force_push must be in [{low:g},{high:g}] m/s')
    friction = values.get('motor_friction')
    friction = 0. if friction is None else friction
    low, high = MOTOR_FRICTION_RANGE
    if type(friction) not in (int, float) or not math.isfinite(friction) or not low <= friction <= high:
        raise ValueError(f'G1 motor_friction must be in [{low:g},{high:g}]')
    if friction and not real:  # only the real-robot profile has the motor model and its friction
        raise ValueError('motor_friction requires real_robot')
    floor = values.get('knee_floor')
    low, high = KNEE_FLOOR_RANGE
    if floor is not None and (type(floor) not in (int, float) or not math.isfinite(floor) or not low <= floor <= high):
        raise ValueError(f'G1 knee_floor must be null or in [{low:g},{high:g}] rad')
    history = values.get('history', 1)
    if type(history) not in (int, float) or history != int(history) or not HISTORY_RANGE[0] <= history <= HISTORY_RANGE[1]:
        raise ValueError(f'G1 history must be a whole number in [{HISTORY_RANGE[0]},{HISTORY_RANGE[1]}]')
    if history > 1 and not real:
        raise ValueError('history requires real_robot')
    if values.get('phase_stop') and not real:
        raise ValueError('phase_stop requires real_robot')
    for key in ('height_command', 'upper_body_slot', 'arm_blind', 'arm_targets'):  # about the deploy observation layout
        if values.get(key) and not real:
            raise ValueError(f'{key} requires real_robot')
    if values.get('arm_targets'):  # the slot carries the targets; the locked wrists keep their offsets at 0
        if not values.get('upper_body_slot') or not values.get('lock_wrists'):
            raise ValueError('arm_targets requires upper_body_slot and lock_wrists')
        if values.get('arm_blind'):
            raise ValueError('arm_targets needs the arms seen: not with arm_blind')
    load_curve = load_curve_setting(values.get('load_curve'))
    if load_curve is not None and not values.get('arm_targets'):
        raise ValueError('load_curve requires arm_targets')
    commands = g1_commands(values.get('commands'))
    if any(commands['crouch_share']) and not values.get('height_command'):
        raise ValueError('G1 command crouch_share requires height_command')
    gait = g1_gait(values.get('gait'))
    if real:  # the robot's controller computes the phase observation with a 0.6 s period
        gait['period'] = DEPLOY_PHASE_PERIOD
    return {'reward_weights': g1_reward_weights(values.get('reward_weights')),
            'heading_hold': values.get('heading_hold', False),
            'commands': commands,
            'gait': gait,
            'action_limit': float(limit),
            'real_robot': real,
            'critic_extras': values.get('critic_extras', False),
            'training_realism': float(realism),
            'history': int(history),
            'stand_heading_hold': values.get('stand_heading_hold', True),
            'phase_stop': values.get('phase_stop', False),
            'sole_contact': values.get('sole_contact', False),
            'random_phase': values.get('random_phase', False),
            'action_delay': float(delay),
            'arm_perturb': float(perturb),
            'arm_perturb_radians': float(radians),
            'arm_perturb_contact_mask': values.get('arm_perturb_contact_mask', False),
            'lock_wrists': values.get('lock_wrists', False),
            'knee_floor': None if floor is None else float(floor),
            'force_push': float(push),
            'feet_flat_stance': values.get('feet_flat_stance', False),
            'foot_collision': values.get('foot_collision', False),
            'side_heading_hold': values.get('side_heading_hold', True),
            'motor_friction': float(friction),
            'height_command': values.get('height_command', False),
            'upper_body_slot': values.get('upper_body_slot', False),
            'arm_blind': values.get('arm_blind', False),
            'hand_collision': values.get('hand_collision', False),
            'arm_targets': values.get('arm_targets', False),
            'load_curve': load_curve}


# Bumped when the real-robot physics or observation layout changes.
REAL_ROBOT_REVISION = 'unitree-g1-velocity-v0: rotor armature, motor curves and friction, deploy observations'


def g1_config_identity(identity, config):
    """Settings that change the policy contract; the original ones keep the original schema names."""
    if config['action_limit'] != 1.:
        identity = {**identity, 'action_schema': f"{identity['task_id']}:action-v2-limit{config['action_limit']:g}"}
    if config.get('lock_wrists'):  # four action entries no longer reach the motors
        identity = {**identity, 'action_schema': f"{identity['action_schema']}-wristlock"}
    if config.get('knee_floor') is not None:  # knee actions below the floor no longer move the knee target
        identity = {**identity, 'action_schema': f"{identity['action_schema']}-knee{config['knee_floor']:g}"}
    if config.get('arm_targets'):  # the arm actions are a residual on the slot's targets
        identity = {**identity, 'action_schema': f"{identity['action_schema']}{ARM_RESIDUAL_SUFFIX}"}
    if config.get('real_robot'):
        import hashlib
        schema = ('obs-unitree-deploy-v0' + (f'-h{config["history"]}' if config.get('history', 1) > 1 else '')
                  + ('-phasestop' if config.get('phase_stop') else '')
                  + ('-hcmd' if config.get('height_command') else '')
                  + ('-ubslot' if config.get('upper_body_slot') else '')
                  + (ARM_BLIND_SUFFIX if config.get('arm_blind') else '')
                  + (ARM_TARGETS_SUFFIX if config.get('arm_targets') else '')
                  + (f'-critic{CRITIC_EXTRAS}' if config.get('critic_extras') else ''))
        identity = {**identity, 'observation_schema': f"{identity['task_id']}:{schema}",
                    'model_revision': hashlib.sha256(f"{identity['model_revision']}:{REAL_ROBOT_REVISION}".encode()).hexdigest()}
    if config.get('foot_collision'):  # contacts added, the same bodies, joints and layout (warm_start.carry_over)
        identity = {**identity, 'model_revision': f"{identity['model_revision']}{FOOT_COLLISION_SUFFIX}"}
    if config.get('hand_collision'):  # likewise
        identity = {**identity, 'model_revision': f"{identity['model_revision']}{HAND_COLLISION_SUFFIX}"}
    return identity


# g1_getup settings (iteration 33): the reward weights (G1_GETUP_REWARD_WEIGHTS), the assist force a fresh run starts
# with (N; each world then loses 20 N whenever it stood up, tasks/g1_tensor_math.assist_update; a resumed run restores
# its worlds' forces from the stage's training_state) and the domain randomization of every episode: the largest
# policy-to-motor delay (s, as task_config action_delay), the largest multiplier of the motor model's joint friction (as
# motor_friction) and the range of one sliding friction drawn for the whole robot and the floor (GETUP_BODY_FRICTION).
# None of them changes the policy contract.
# pd_bound (iteration 34, HoST's β): a hard bound on the PD error, [start, lowest] in shares c of each motor's peak
# torque y1: every motor target but the locked wrists' stays within c·y1/kp of the joint angle the policy observed
# (tasks/g1_tensor_math.pd_bound), so the active torque is at most c·y1. Each world starts at `start` and steps
# PD_BOUND_STEP down to `lowest` at a reset after an episode up, by the assist's rule (pd_bound_update); stages keep
# the worlds' c (training_state), evaluation and export use `lowest`. A different meaning of the actions: its own
# action contract (action_schema …-pdbound<lowest>, g1_getup_config_identity), a stage moves there with
# warm_start.carry_over. None (missing) is the original task, bit for bit.
# stand_pose_std (iteration 36, rad): the width σ of the standing pose term w·[standing]·exp(−mean((q − home)²)/σ²)
# (tasks/g1_tensor_math.getup_reward_components). At GETUP_POSE_STD (missing), the original 0.1, the term is dead after
# getting up (critic, lab-critic/it36/post_pose.txt: the arms, waist and right shoulder yaw rest on their stops, Σ Δq²
# ≈ 23 rad², e^−83; candidate 8 standing 0.18); at the working value 0.7 it gives 0.09-0.10 per standing step there and
# 0.494 in candidate 8's pose. A reward setting only: the contract and branching are unchanged.
# action_change_unbounded (iteration 37, bool): the action_change term −w·Σ(a − a_prev)² charges the action before
# pd_bound (the box-clipped one with the locked wrists at 0) and that action of the step before, kept on its own (zero
# at a reset, as the applied one), instead of the applied, bounded action. Why (critic, lab-critic/it37, noise_cost.py,
# noise_episode.py): an action mean beyond pd_bound's clamp hides the policy's noise from the term, so PPO pushes the
# mean out. After getting up the right shoulder yaw's mean stands at 8.42, its range edge at 5.95 and the clamp's at
# 8.32 (σ 1.0); its noise costs 0.0082 per step there against 0.0189 at the range edge. The clamp hides 38 % of the
# joint-steps (before getting up 0.279 per step against 0.574 unclamped), and at the hand-over to walking that arm
# swings at up to 29 rad/s. The PD, the delay queue and the observed last action keep the applied action, PPO the
# policy's. A reward setting only: the contract and branching are unchanged. False (missing) is the original term, bit
# for bit; without a bound the two actions are the same.
# style_library (iteration 53): the name of the style term's human frame library, the file
# runs/humanoid_lab/getup_style/<name>.npz (tasks/g1_getup_style; scripts/lab/getup_style_library.py makes it; LAFAN1
# is CC BY-NC-ND 4.0, so the library stays out of the repository). Read only while the style weight is non-zero, so
# the default name changes nothing for a task without that weight.
# style_judge (iteration 55, bool): the full-GPU task computes rollout/style_dist and rollout/style_band (the distance
# to the style library) without paying the style term, an independent judge of a run whose style weight is 0; on by
# itself while the amp weight is non-zero. False (missing) with both weights at 0 reads nothing, as before.
# human_starts (iteration 56, reference state init): {share, md5}, a share of training resets starts from a human frame
# getting up (tasks/g1_getup_human: runs/humanoid_lab/getup_bank/human_starts.npz, its md5) instead of a bank row;
# their episodes do not step the assist or pd_bound curricula. Evaluation, probes and held-out bank rows are unchanged.
# None (missing) is the task before it, bit for bit. Not part of the contract.
# stance_pelvis (iteration 60, m; the critic's lab-critic/it53/split59.py): the stance_width term counts with the pelvis
# above it (and both soles down). At GETUP_STANCE_PELVIS (missing), the original 0.5, a split below it is not charged:
# stage 1164 jumps out of one, the feet 1.01-1.10 m apart before the jump, 0.31-0.41 m after. Working value 0.25.
GETUP_STANCE_PELVIS = .5
GETUP_STANCE_PELVIS_RANGE = (.1, .8)
# hip_roll_out (iteration 63, rad; the critic's lab-critic/it63, pose63.py, hum63.py): the hip roll PD target's limit
# outwards, per leg (the left leg's +, the right leg's −, g1_getup.outward_signs), through target_low / target_high
# as knee_floor: the PD, the delay queue and arm_press; the action and the observed last action are not changed.
# Why: 1164's "spider", the hips spread in roll by 2.0-2.4 rad (p90 3.5) from the front to standing, the feet
# 1.2-1.4 m apart; a human's spread over 100 LAFAN1 rises median 0.88, p90 1.46 (its feet 0.50 m). Its own action
# contract (…-hiproll<v>, g1_getup_config_identity), a stage moves there with warm_start.carry_over. None (missing):
# off, bit for bit. Getting up only: walking keeps its joint ranges.
GETUP_HIP_ROLL_RANGE = (.2, 2.5)
# track_clip (iteration 64; the user's decision: the robot gets up as a human does, not as a "spider"): {file, md5}, the
# motion tracking mode of g1_getup, BeyondMimic / mjlab style: the policy follows one human clip getting up
# (tasks/g1_getup_track: runs/humanoid_lab/getup_track/<file>.npz, made by scripts/lab/getup_track_clip.py from LAFAN1,
# CC BY-NC-ND 4.0: a research prototype, never published, scripts/robot/package.py refuses such a stage), then holds
# its last frame for TRACK_HOLD_SECONDS. The getting-up terms (head, upright, stand…, head_rise, stance, style, amp)
# and the assist are off; the reward is the tracking terms of G1_GETUP_TRACK_WEIGHTS (g1_tensor_math.
# track_reward_components) with the impact and sole_impact weights of reward_weights. The actor's 490 inputs, the
# actions and the export are unchanged; the critic gets the clip's phase and the anchor's errors after the six extras
# (TRACK_CRITIC_EXTRAS more: its own observation contract, …-critic6-track11, g1_getup_config_identity). Not with
# pd_bound, human_starts, style_judge or the style / amp weights. None (missing): off, the task bit for bit.
GETUP_TRACK_NAME = r'[A-Za-z0-9_-]{1,64}'  # a file name under getup_track, never a path
TRACK_CRITIC_EXTRAS = 11  # sin/cos of the clip phase, the anchor's position (3) and orientation (6) error
# Tracking weights (mjlab / BeyondMimic G1 tracking; exp(−e/σ²) terms, σ in g1_tensor_math.TRACK_SIGMA): the anchor
# (torso_link) position and orientation, the 14 bodies' positions relative to the anchor, orientations, linear and
# angular velocities; penalties: action rate, joints beyond their soft limits (rad), self_contact on our scale (the
# share of substeps with a robot-robot contact above 10 N).
G1_GETUP_TRACK_WEIGHTS = dict(anchor_pos=.5, anchor_ori=.5, body_pos=1., body_ori=1., body_lin_vel=1., body_ang_vel=1.,
                              action_rate=.1, joint_limit=10., self_contact=10.)
# settle_clip (iteration 66; the user: get up from any fall pose, as a human does): [{file, md5}, …], one to
# SETTLE_CLIPS_MAX human clips getting up (tasks/g1_getup_track's files, as track_clip's) whose first frames are the
# settling targets: from any bank start (g1_tensor_math.SETTLE_CLASSES) the policy rolls and lays its limbs into the
# nearest of these start poses (the back's cmu-140_09, the front's cmu-140_01), where a tracker of that clip takes
# over. The reward (g1_tensor_math.settle_reward_components, G1_GETUP_SETTLE_WEIGHTS with reward_weights' impact and
# sole_impact): the pelvis gravity and the free joints against the nearest target, stillness once near it,
# regularization; success (the hand-over to g1_ctrl's tracker) the pose within SETTLE_DONE_ANGLE / SETTLE_DONE_POSE,
# every free joint slower than SETTLE_DONE_SPEED for SETTLE_DONE_SECONDS. Episodes SETTLE_EPISODE_SECONDS, no assist.
# The actor's 490 inputs, the critic's six extras, the actions and the export are unchanged: not part of the contract.
# Not with track_clip, pd_bound, human_starts, style_judge or the style / amp weights. None (missing): off, bit for bit.
SETTLE_CLIPS_MAX = 4
# Settling weights: the orientation and pose kernels (σ in g1_tensor_math.SETTLE_ORI_STD / SETTLE_POSE_STD), the
# stillness penalty near the target, action rate as getting up's action_change, the joints beyond their soft limits
# (all joints, rad), self_contact on our scale (the share of substeps with a robot-robot contact above 10 N).
G1_GETUP_SETTLE_WEIGHTS = dict(settle_ori=1., settle_pose=1., settle_still=1., action_rate=.01, joint_limit=1.,
                               self_contact=1.)
GETUP_POSE_STD = .1
GETUP_POSE_STD_RANGE = (.02, 2.)
GETUP_STYLE_LIBRARY = 'lafan1'
GETUP_STYLE_NAME = r'[A-Za-z0-9_-]{1,64}'  # a file name under getup_style, never a path
G1_GETUP_DEFAULTS = dict(assist_force=200., action_delay=.02, motor_friction=1.25, body_friction=[.3, 1.],
                         pd_bound=None, stand_pose_std=GETUP_POSE_STD, action_change_unbounded=False,
                         style_library=GETUP_STYLE_LIBRARY, style_judge=False, human_starts=None,
                         stance_pelvis=GETUP_STANCE_PELVIS, hip_roll_out=None, track_clip=None, settle_clip=None)
GETUP_ASSIST_RANGE = (0., 500.)
GETUP_BODY_FRICTION_RANGE = (.05, 2.)
PD_BOUND_RANGE = (.1, 1.)
G1_GETUP_LABELS = dict(head='Высота макушки: от 0,15 до 1,20 м', upright='Корпус вертикально (при тазе выше 0,4 м)',
                       stand='Бонус за стойку: макушка выше 1,15 м, наклон меньше 15°, таз выше 0,70 м, обе стопы на полу',
                       stand_pose='Поза стоя близко к исходной (только в стойке)',
                       stand_still='Штраф за скорость таза в стойке', effort='Штраф за моменты моторов',
                       joint_speed='Штраф за скорости суставов', action_change='Штраф за резкие действия',
                       joint_limits='Штраф за ноги у упоров (последние 5% хода бедра, колена, голеностопа)',
                       impact=('Штраф за удары о пол не стопами: сумма по телам квадратов превышения скорости '
                               'сближения с полом над 1,5 м/с'),
                       sole_impact=('Штраф за жёсткую постановку стоп (приземление после прыжка): сумма по стопам '
                                    'квадратов превышения скорости сближения подошвы с полом над 0,5 м/с'),
                       stance_width=('Штраф за широкую стойку: на сколько метров стопы шире 0,30 м поперёк курса таза '
                                     '(при тазе выше 0,5 м и обеих подошвах на полу)'),
                       arm_press=('Штраф за давление рук стоя: сумма по суставам рук без закреплённых запястий '
                                  '(цель мотора до границы мягкости − угол)², рад² (только в стойке)'),
                       style=('Награда за стиль: exp(−d/2), d — расстояние до ближайшего кадра вставания человека '
                              '(поза и скорости суставов, гравитация и высота таза), пока макушка на 0,3–1,1 м'),
                       head_rise=('Штраф за слишком быстрый подъём головы: на сколько м/с макушка поднимается '
                                  'быстрее 0,8 м/с (на каждом шаге)'),
                       sole_slip=('Штраф за скольжение стоп: сумма по стопам на полу (подошва ниже 1 см) квадрата '
                                  'скорости стопы по полу, м²/с²'),
                       amp=('Награда за стиль от дискриминатора AMP: max(0, 1 − 0,25·(D − 1)²) по паре соседних '
                            'состояний, пока макушка на 0,3–1,1 м (только полный GPU; первые 5 млн шагов — прогрев)'))
G1_GETUP_PARAMETERS = (
    dict(group=None, key='assist_force', default=G1_GETUP_DEFAULTS['assist_force'], min=GETUP_ASSIST_RANGE[0],
         max=GETUP_ASSIST_RANGE[1],
         label='Подтяжка за таз вверх в начале обучения, Н (у мира, который встал, убывает на 20 Н)'),
    dict(group=None, key='action_delay', default=G1_GETUP_DEFAULTS['action_delay'], min=ACTION_DELAY_RANGE[0],
         max=ACTION_DELAY_RANGE[1], label='Случайная задержка команды в обучении, с (0 — без задержки)'),
    dict(group=None, key='motor_friction', default=G1_GETUP_DEFAULTS['motor_friction'], min=MOTOR_FRICTION_RANGE[0],
         max=MOTOR_FRICTION_RANGE[1],
         label=('Разброс трения моторов в обучении: верхняя граница множителя трения суставов (0 — без разброса; '
                'на эпизод множитель от 0 до этого числа)')),
    dict(group=None, key='body_friction', default=G1_GETUP_DEFAULTS['body_friction'],
         min=GETUP_BODY_FRICTION_RANGE[0], max=GETUP_BODY_FRICTION_RANGE[1],
         label='Трение тела и пола на эпизод: от и до, пара чисел'),
    # Missing or null in a saved config means off (the motor target anywhere in the box); a different action contract.
    dict(group=None, key='pd_bound', default=G1_GETUP_DEFAULTS['pd_bound'], min=PD_BOUND_RANGE[0],
         max=PD_BOUND_RANGE[1],
         label=('Мягкость: цель мотора не дальше доли c пика момента от угла сустава; пара [начало, минимум], у мира, '
                'который встал, c убывает на 0,02 (пусто — без границы)')),
    # Missing in a saved config means GETUP_POSE_STD (the original pose term).
    dict(group=None, key='stand_pose_std', default=G1_GETUP_DEFAULTS['stand_pose_std'], min=GETUP_POSE_STD_RANGE[0],
         max=GETUP_POSE_STD_RANGE[1],
         label=('Поза стоя: ширина σ, рад, награды exp(−среднее по суставам (q − исходная)²/σ²) '
                '(0,1 — прежняя; шире — тянет к исходной позе и издалека)')),
    # Missing means off (the term charged the applied, bounded action).
    dict(group=None, key='action_change_unbounded', default=G1_GETUP_DEFAULTS['action_change_unbounded'],
         type='bool',
         label=('Штраф за резкие действия — по действию до границы мягкости (а не по ограниченному: граница '
                'не прячет шум)')),
    # Read only while the style weight is non-zero.
    dict(group=None, key='style_library', default=G1_GETUP_DEFAULTS['style_library'], type='text',
         label='Библиотека кадров человека для награды за стиль (файл runs/humanoid_lab/getup_style/<имя>.npz)'),
    # Missing means off; the amp weight turns the judge on by itself.
    dict(group=None, key='style_judge', default=G1_GETUP_DEFAULTS['style_judge'], type='bool',
         label='Считать расстояние до кадров человека в журнале обучения без награды за стиль (судья)'),
    # Missing or null means off (every training episode starts from the bank).
    dict(group=None, key='human_starts', default=G1_GETUP_DEFAULTS['human_starts'], type='json',
         label=('Старты из кадров вставания человека: {"share": доля обучающих сбросов, "md5": md5 файла '
                'runs/humanoid_lab/getup_bank/human_starts.npz} (пусто — все старты из банка)')),
    # Missing means GETUP_STANCE_PELVIS (the original stance_width term).
    dict(group=None, key='stance_pelvis', default=G1_GETUP_DEFAULTS['stance_pelvis'],
         min=GETUP_STANCE_PELVIS_RANGE[0], max=GETUP_STANCE_PELVIS_RANGE[1],
         label='Штраф за широкую стойку действует, когда таз выше этой высоты, м (0,5 — прежний)'),
    # Missing or null means off (the hip roll target anywhere in its joint range); a different action contract.
    dict(group=None, key='hip_roll_out', default=G1_GETUP_DEFAULTS['hip_roll_out'], min=GETUP_HIP_ROLL_RANGE[0],
         max=GETUP_HIP_ROLL_RANGE[1],
         label='Предел цели крена бедра наружу, рад, на каждую ногу (пусто — без предела)'),
    # Missing or null means off (the getting-up task); a different critic contract.
    dict(group=None, key='track_clip', default=G1_GETUP_DEFAULTS['track_clip'], type='json',
         label=('Следование записи человека: {"file": имя клипа runs/humanoid_lab/getup_track/<имя>.npz, "md5": md5 '
                'файла} (пусто — обычная задача вставания)')),
    # Missing or null means off (the getting-up task); not part of the contract.
    dict(group=None, key='settle_clip', default=G1_GETUP_DEFAULTS['settle_clip'], type='json',
         label=('Укладка в начальную позу клипа вставания: [{"file": имя клипа runs/humanoid_lab/getup_track/<имя>.npz, '
                '"md5": md5 файла}, …], цель — ближайший из первых кадров (пусто — обычная задача вставания)')),
    *(dict(group='reward_weights', key=key, label=G1_GETUP_LABELS[key], default=G1_GETUP_REWARD_WEIGHTS[key], min=0,
           max=100) for key in G1_GETUP_REWARD_WEIGHTS),
)


def g1_getup_config(values):
    """g1_getup's task_config, every setting spelled out (G1_GETUP_DEFAULTS for the missing ones; pd_bound None:
    off; stand_pose_std GETUP_POSE_STD: the original pose term; action_change_unbounded false: the original
    action_change term)."""
    if not isinstance(values, dict) or set(values) - {'reward_weights', *G1_GETUP_DEFAULTS}:
        raise ValueError('Unknown G1 getup setting')
    weights = values.get('reward_weights') or {}
    if not isinstance(weights, dict) or set(weights) - set(G1_GETUP_REWARD_WEIGHTS):
        raise ValueError('Unknown G1 getup reward weight')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100 for v in weights.values()):
        raise ValueError('Reward weights must be finite and in [0,100]')
    result = {**G1_GETUP_DEFAULTS, **{key: value for key, value in values.items() if key != 'reward_weights'}}
    for key, (low, high) in (('assist_force', GETUP_ASSIST_RANGE), ('action_delay', ACTION_DELAY_RANGE),
                             ('motor_friction', MOTOR_FRICTION_RANGE), ('stand_pose_std', GETUP_POSE_STD_RANGE),
                             ('stance_pelvis', GETUP_STANCE_PELVIS_RANGE)):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'G1 getup {key} must be in [{low:g},{high:g}]')
    friction, (low, high) = result['body_friction'], GETUP_BODY_FRICTION_RANGE
    if not isinstance(friction, (list, tuple)) or len(friction) != 2 or any(
            type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high for v in friction) \
            or friction[0] > friction[1]:
        raise ValueError(f'G1 getup body_friction must be two numbers (from, to) in [{low:g},{high:g}]')
    bound, (low, high) = result['pd_bound'], PD_BOUND_RANGE
    if bound is not None and (not isinstance(bound, (list, tuple)) or len(bound) != 2 or any(
            type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high for v in bound)
            or bound[0] < bound[1]):
        raise ValueError(f'G1 getup pd_bound must be null or two numbers (start, lowest) in [{low:g},{high:g}], '
                         'start >= lowest')
    if type(result['action_change_unbounded']) is not bool:
        raise ValueError('G1 getup action_change_unbounded must be true or false')
    if not isinstance(result['style_library'], str) or not re.fullmatch(GETUP_STYLE_NAME, result['style_library']):
        raise ValueError('G1 getup style_library must be a library name (letters, digits, _ and -)')
    if type(result['style_judge']) is not bool:
        raise ValueError('G1 getup style_judge must be true or false')
    human = result['human_starts']
    if human is not None:
        share = human.get('share') if isinstance(human, dict) else None
        if not isinstance(human, dict) or set(human) != {'share', 'md5'} or type(share) not in (int, float) \
                or not math.isfinite(share) or not 0 < share <= 1 or not isinstance(human['md5'], str) \
                or not re.fullmatch(r'[0-9a-f]{32}', human['md5']):
            raise ValueError('G1 getup human_starts must be null or {"share": (0, 1], "md5": the file\'s md5}')
        human = {'share': float(share), 'md5': human['md5']}
    roll, (low, high) = result['hip_roll_out'], GETUP_HIP_ROLL_RANGE
    if roll is not None and (type(roll) not in (int, float) or not math.isfinite(roll) or not low <= roll <= high):
        raise ValueError(f'G1 getup hip_roll_out must be null or in [{low:g},{high:g}] rad')
    clip = result['track_clip']
    if clip is not None:
        if not isinstance(clip, dict) or set(clip) != {'file', 'md5'} or not isinstance(clip['file'], str) \
                or not re.fullmatch(GETUP_TRACK_NAME, clip['file']) or not isinstance(clip['md5'], str) \
                or not re.fullmatch(r'[0-9a-f]{32}', clip['md5']):
            raise ValueError('G1 getup track_clip must be null or {"file": a clip name (letters, digits, _ and -), '
                             '"md5": the file\'s md5}')
        if bound is not None or human is not None or result['style_judge'] or any(
                weights.get(key) for key in ('style', 'amp')):
            raise ValueError('G1 getup track_clip does not go with pd_bound, human_starts, style_judge or the style '
                             'and amp weights')
        clip = {'file': clip['file'], 'md5': clip['md5']}
    settle = result['settle_clip']
    if settle is not None:
        if not isinstance(settle, (list, tuple)) or not 1 <= len(settle) <= SETTLE_CLIPS_MAX or any(
                not isinstance(entry, dict) or set(entry) != {'file', 'md5'} or not isinstance(entry['file'], str)
                or not re.fullmatch(GETUP_TRACK_NAME, entry['file']) or not isinstance(entry['md5'], str)
                or not re.fullmatch(r'[0-9a-f]{32}', entry['md5']) for entry in settle) \
                or len({entry['md5'] for entry in settle}) != len(settle):
            raise ValueError(f'G1 getup settle_clip must be null or a list of 1 to {SETTLE_CLIPS_MAX} different '
                             '{"file": a clip name (letters, digits, _ and -), "md5": the file\'s md5}')
        if clip is not None or bound is not None or human is not None or result['style_judge'] or any(
                weights.get(key) for key in ('style', 'amp')):
            raise ValueError('G1 getup settle_clip does not go with track_clip, pd_bound, human_starts, style_judge '
                             'or the style and amp weights')
        settle = [{'file': entry['file'], 'md5': entry['md5']} for entry in settle]
    return {'reward_weights': {**G1_GETUP_REWARD_WEIGHTS, **{k: float(v) for k, v in weights.items()}},
            'assist_force': float(result['assist_force']), 'action_delay': float(result['action_delay']),
            'motor_friction': float(result['motor_friction']), 'body_friction': [float(v) for v in friction],
            'pd_bound': None if bound is None else [float(v) for v in bound],
            'stand_pose_std': float(result['stand_pose_std']),
            'action_change_unbounded': result['action_change_unbounded'], 'style_library': result['style_library'],
            'style_judge': result['style_judge'], 'human_starts': human,
            'stance_pelvis': float(result['stance_pelvis']), 'hip_roll_out': None if roll is None else float(roll),
            'track_clip': clip, 'settle_clip': settle}


def g1_getup_config_identity(identity, config):
    """g1_getup's contract refined by its normalized task_config (TaskSpec.config_identity): pd_bound bounds what the
    actions do, its own action contract (…-pdbound<lowest>: g1_getup:action-v2-limit10-wristlock-knee0.061-pdbound0.6
    for [1, 0.6]), and so does hip_roll_out (…-hiproll<v>, after it); track_clip adds TRACK_CRITIC_EXTRAS critic inputs,
    its own observation contract (…-critic6-track11; the actor's inputs as before); without them the identity of
    g1_getup_identity, bit for bit."""
    if config.get('pd_bound') is not None:
        identity = {**identity, 'action_schema': f"{identity['action_schema']}-pdbound{config['pd_bound'][1]:g}"}
    if config.get('hip_roll_out') is not None:  # hip roll actions beyond the limit no longer move the target
        identity = {**identity, 'action_schema': f"{identity['action_schema']}-hiproll{config['hip_roll_out']:g}"}
    if config.get('track_clip') is not None:  # the critic's inputs grow; the actor's stay
        identity = {**identity, 'observation_schema': f"{identity['observation_schema']}-track{TRACK_CRITIC_EXTRAS}"}
    return identity


def g1_getup_identity(identity):
    """g1_getup's contract from its asset identity (task_id g1_getup): G1_GETUP_PROFILE's model revision, action schema
    and observation schema as g1_config_identity gives them, the critic extras counted as GETUP_CRITIC_EXTRAS
    (g1_getup:obs-unitree-deploy-v0-h5-phasestop-critic6, g1_getup:action-v2-limit10-wristlock-knee0.061)."""
    identity = g1_config_identity(identity, g1_config(G1_GETUP_PROFILE))
    schema = identity['observation_schema']
    if not schema.endswith(f'-critic{CRITIC_EXTRAS}'):
        raise ValueError('g1_getup expects the walking layout with critic extras')
    return {**identity, 'observation_schema': schema[:-len(f'-critic{CRITIC_EXTRAS}')] + f'-critic{GETUP_CRITIC_EXTRAS}'}


def empty_config(values):
    if not isinstance(values, dict) or values:
        raise ValueError('This task has no configurable parameters')
    return {}
