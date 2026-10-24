"""Bounded task parameters; callers never supply arbitrary environment kwargs."""
import math

# g1_velocity v2 weights. A key missing from a saved config takes this value, so terms added later
# default to "absent" (forward, sideways, torso, orientation, height = 0) and old checkpoints keep their reward.
G1_REWARD_WEIGHTS = dict(velocity=2., forward=0., sideways=0., heading=0., yaw=1., yaw_drift=0., alive=.5, gait=.5,
                         swing=10., tilt=1.,
                         orientation=0., height=0., height_ceiling=0., torso=0., base_motion=2., still=0.,
                         stand_pose=0., slip=0., feet_flat=0., clearance=0., self_contact=0., arm_press=0., posture=.1,
                         hips=0., effort=1e-5, action_change=.01, fall=5.)
# Values the browser form starts with for a new experiment.
G1_RECOMMENDED = {**G1_REWARD_WEIGHTS, 'torso': 2.}
# g1_getup weights (iteration 33, the critic's specification; tasks/g1_tensor_math.getup_reward_components): the task
# (head height 1, torso upright 0.5), the standing state (bonus 1, pose 0.5, pelvis stillness), regularization always
# (torques 2e-5, joint speeds 1e-4, action change as the walking recipe's 0.05, legs beyond the soft joint limits).
# stand_still and joint_limits were not given numbers: 1 each, to be tuned by branches.
G1_GETUP_REWARD_WEIGHTS = dict(head=1., upright=.5, stand=1., stand_pose=.5, stand_still=1., effort=2e-5,
                               joint_speed=1e-4, action_change=.05, joint_limits=1.)
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
    (the hands touched nothing).
    """
    keys = {'reward_weights', 'heading_hold', 'commands', 'gait', 'action_limit', 'real_robot', 'critic_extras',
            'training_realism', 'history', 'stand_heading_hold', 'phase_stop', 'sole_contact', 'random_phase',
            'action_delay', 'arm_perturb', 'arm_perturb_radians', 'arm_perturb_contact_mask', 'lock_wrists',
            'knee_floor', 'force_push', 'feet_flat_stance', 'foot_collision', 'side_heading_hold', 'motor_friction',
            'height_command', 'upper_body_slot', 'arm_blind', 'hand_collision'}
    if not isinstance(values, dict) or set(values)-keys:
        raise ValueError('Unknown G1 task setting')
    for key in ('heading_hold', 'real_robot', 'critic_extras', 'stand_heading_hold', 'phase_stop', 'sole_contact',
                'random_phase', 'arm_perturb_contact_mask', 'lock_wrists', 'feet_flat_stance', 'foot_collision',
                'side_heading_hold', 'height_command', 'upper_body_slot', 'arm_blind', 'hand_collision'):
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
    for key in ('height_command', 'upper_body_slot', 'arm_blind'):  # about the deploy observation layout
        if values.get(key) and not real:
            raise ValueError(f'{key} requires real_robot')
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
            'hand_collision': values.get('hand_collision', False)}


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
    if config.get('real_robot'):
        import hashlib
        schema = ('obs-unitree-deploy-v0' + (f'-h{config["history"]}' if config.get('history', 1) > 1 else '')
                  + ('-phasestop' if config.get('phase_stop') else '')
                  + ('-hcmd' if config.get('height_command') else '')
                  + ('-ubslot' if config.get('upper_body_slot') else '')
                  + (ARM_BLIND_SUFFIX if config.get('arm_blind') else '')
                  + (f'-critic{CRITIC_EXTRAS}' if config.get('critic_extras') else ''))
        identity = {**identity, 'observation_schema': f"{identity['task_id']}:{schema}",
                    'model_revision': hashlib.sha256(f"{identity['model_revision']}:{REAL_ROBOT_REVISION}".encode()).hexdigest()}
    if config.get('foot_collision'):  # contacts added, the same bodies, joints and layout (warm_start.carry_over)
        identity = {**identity, 'model_revision': f"{identity['model_revision']}{FOOT_COLLISION_SUFFIX}"}
    if config.get('hand_collision'):  # likewise
        identity = {**identity, 'model_revision': f"{identity['model_revision']}{HAND_COLLISION_SUFFIX}"}
    return identity


def empty_config(values):
    if not isinstance(values, dict) or values:
        raise ValueError('This task has no configurable parameters')
    return {}
