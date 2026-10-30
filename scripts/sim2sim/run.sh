#!/bin/bash
# One sim2sim run: unitree_mujoco (G1 29-DoF, band) + g1_ctrl --network=lo + virtual gamepad scenario.
#
# Run as root from Windows (PowerShell), after setup.sh:
#   wsl.exe -d Ubuntu -u root -e bash /mnt/c/Users/Admin/Documents/Codex/Sim/scripts/sim2sim/run.sh [export dir] [gamepad.py options]
# export dir: output of `python -m humanoid_sim.export_g1` (policy.onnx + deploy.yaml), as a WSL path.
# gamepad.py options: --scenario walk|turn|side|sidestop|stopstart|passive|stand|push|crouch|arms (default walk), --walk,
#   --yaw, --side, --stand-seconds, --push-walking, --push-strength weak|strong|both, --push-dv 1.25,1.5,
#   --push-order, --ry, --poses, --pose-seconds, --arm-walk ... (push needs setup.sh's unitree_mujoco push patch;
#   crouch and --ry a height export, task_config height_command, and setup.sh's g1_ctrl-height.patch: the right stick
#   down is the height command; arms an arm_targets export and g1_ctrl-arms.patch: its poses by LB + a button, the
#   export's deploy.yaml goes to gamepad.py as --deploy)
# run.sh options:
#   --fixstand-wrist left|right[=rad]  FixStand ends with that wrist pitch (SDK 20 left, 27 right) at this target
#                                      instead of Unitree's 0 (default 0.699999988: candidate 2's own target for
#                                      its bent wrist, docs/sim2sim.md «Режим запястий»); the other wrist and
#                                      joints keep Unitree's pose. g1_ctrl reads config/config.yaml next to its
#                                      build dir, so the run edits that file (fixstand.py) and puts the original
#                                      back when it ends; without the option the file is not touched.
#   --fixstand-left-wrist[=rad]        the same as --fixstand-wrist left[=rad] (its first name)
#   --getup <export dir>               a g1_getup export (export_g1 of a getting-up stage) for g1_ctrl's GetUp state
#                                      (setup.sh's g1_ctrl-getup.patch): installed into config/policy/getup and the
#                                      state turned on in config/config.yaml for this run (the original put back when
#                                      it ends, as with --fixstand-wrist); --scenario getup falls, lies and enters
#                                      it (gamepad.py --fall, --getup-velocity: then the first argument's walking
#                                      export in Velocity). The getup scenario's report is g1_ctrl's state changes
#                                      and getup_report.py (getup.json), no analyze.py.
#   --getup-track <export dir>         with --getup: the getting-up chain (setup.sh's g1_ctrl-getupchain.patch,
#                                      scripts/lab/probes/getup_chain.py): a tracker export (export_g1 of a
#                                      track_clip stage with --handover-frame: its handover.yaml) installed into
#                                      config/policy/getup_track and GetUpBlend / GetUpTrack turned on in config.yaml
#                                      for this run, so GetUp hands over by itself (the pelvis rule), the PD takes it
#                                      to the tracker's frame and the tracker runs; gamepad.py waits for GetUpTrack in
#                                      g1_ctrl.log and presses RT + A --track-seconds later (default: the clip's rest
#                                      from the frame plus TRACK_RT_HOLD s). The run's folder gets -chain.
#   --hand-load <kg>[,<kg>]            loads in the hands of unitree_mujoco's G1 (both, or left,right), as
#                                      task_config arm_targets trains them: hand_load.py writes scene_29dof_load.xml
#                                      (each wrist_yaw_link's mass + kg, its COM moved toward the palm, x 0.11 in the
#                                      link) next to Unitree's and the run puts it into simulate/config.yaml's
#                                      robot_scene (the original put back when it ends, as with --fixstand-wrist).
#   --hand-capsules                    Unitree's hand capsules (task_config hand_collision) on unitree_mujoco's G1,
#                                      whose rubber hands do not collide: hand_load.py --capsules, the same scene as
#                                      --hand-load (load 0 unless given). The run's folder gets -caps.
# Results: $ROOT/sim2sim_logs/<export>-<time>[-<scenario>[<side>|<fall>][-walking][-dv<v>[_<v>...]]]
#   [-leftwrist|-rightwrist][-load<kg>][-caps]/ (CSV logs, process logs, g1_ctrl_config.yaml as used, report.txt /
#   report.json);
#   <side> is --side's value for side / sidestop when given (side0.5, sidestop-0.3), <fall> getup's --fall,
#   --fall-push and --fall-passive when given (-stand-back2.25-limp, -forward1).
set -u
export LC_ALL=C

ROOT=/home/mawgly/unitree
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT=${1:-/mnt/c/Users/Admin/Documents/Codex/Sim/runs/humanoid_lab/export/exp-e4da6963dbc7432b-5300944896}
shift $(( $# > 0 ? 1 : 0 ))
# float32(0.07 * 10): wrist pitch action scale times the action limit, the target in targets_sim.csv
WRIST_DEFAULT=0.699999988
WRIST=""  # left|right[=rad]
GETUP=""  # a g1_getup export dir
TRACK=""  # a tracker export dir (the getting-up chain)
TRACK_RT_HOLD=1.5  # s after the clip's end (the tracker holds the last pose 3 s) before RT + A
LOAD=""  # hand loads, kg (one number or left,right)
CAPSULES=""  # --hand-capsules: Unitree's hand capsules on unitree_mujoco's G1 (hand_load.py --capsules)
ARGS=()  # gamepad.py's options
while [ $# -gt 0 ]; do
  case $1 in
    --getup) GETUP=${2:-}; shift ;;
    --getup=*) GETUP=${1#*=} ;;
    --getup-track) TRACK=${2:-}; shift ;;
    --getup-track=*) TRACK=${1#*=} ;;
    --hand-load) LOAD=${2:-}; shift ;;
    --hand-capsules) CAPSULES=--capsules ;;
    --hand-load=*) LOAD=${1#*=} ;;
    --fixstand-wrist) WRIST=${2:-}; shift ;;
    --fixstand-wrist=*) WRIST=${1#*=} ;;
    --fixstand-left-wrist) WRIST=left ;;
    --fixstand-left-wrist=*) WRIST=left=${1#*=} ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
WRIST_SIDE=${WRIST%%=*}
WRIST_Q=$WRIST_DEFAULT
[[ $WRIST == *=* ]] && WRIST_Q=${WRIST#*=}
case $WRIST_SIDE in
  left) WRIST_JOINT=20 ;;
  right) WRIST_JOINT=27 ;;
  "") ;;
  *) echo "--fixstand-wrist: left or right, not '$WRIST_SIDE'"; exit 1 ;;
esac
SCENARIO=""
ARM_SCENARIO=""
WALKING=""
DV=""
SIDE=""
FALL=""
for i in "${!ARGS[@]}"; do
  [ "${ARGS[$i]}" = --scenario ] && SCENARIO=-${ARGS[$((i + 1))]:-}
  [ "${ARGS[$i]}" = --push-walking ] && WALKING=-walking
  [ "${ARGS[$i]}" = --push-dv ] && DV=-dv${ARGS[$((i + 1))]:-}
  [ "${ARGS[$i]}" = --side ] && SIDE=${ARGS[$((i + 1))]:-}
  [ "${ARGS[$i]}" = --fall ] && FALL=-${ARGS[$((i + 1))]:-}$FALL
  [ "${ARGS[$i]}" = --fall-push ] && FALL=$FALL-${ARGS[$((i + 1))]/:/}
  [ "${ARGS[$i]}" = --fall-passive ] && FALL=$FALL-limp
  [ "${ARGS[$i]}" = --scenario ] && [ "${ARGS[$((i + 1))]:-}" = arms ] && ARM_SCENARIO=yes
done
[ "$SCENARIO" = -walk ] && SCENARIO=""
[[ $SCENARIO == -side || $SCENARIO == -sidestop ]] && SCENARIO=$SCENARIO$SIDE
[ "$SCENARIO" = -getup ] && SCENARIO=$SCENARIO$FALL
SCENARIO=$SCENARIO$WALKING${DV//,/_}
TAG=""
[ -n "$WRIST_SIDE" ] && TAG=-${WRIST_SIDE}wrist
[ -n "$LOAD" ] && TAG=$TAG-load${LOAD//,/_}
[ -n "$CAPSULES" ] && TAG=$TAG-caps
[ -n "$TRACK" ] && TAG=$TAG-chain
NAME=$(basename "$EXPORT")-$(date +%Y%m%d-%H%M%S)$SCENARIO$TAG
OUT=$ROOT/sim2sim_logs/$NAME
FIFO=/tmp/sim2sim_js0
G1=$ROOT/unitree_rl_mjlab/deploy/robots/g1
CFG=$G1/config/config.yaml
# g1_ctrl takes the last folder (sorted) under config/policy/velocity that has `exported`
# (param::parser_policy_dir); v9_lab sorts after their v0.
POLICY=$G1/config/policy/velocity/v9_lab
AS_USER=(setpriv --reuid=mawgly --regid=mawgly --init-groups
  env -u WAYLAND_DISPLAY HOME=/home/mawgly USER=mawgly DISPLAY="${DISPLAY:-:0}")  # X11 for xdotool
# Software OpenGL by default: WSLg's GPU path stalled the whole simulator (DDS bridge included) for
# ~17 ms ten times in 10 s; llvmpipe keeps gaps under ~10 ms and leaves the GPU to training.
# SIM2SIM_GL=gpu renders on the GPU instead.
GL=(LIBGL_ALWAYS_SOFTWARE=1)
[ "${SIM2SIM_GL:-soft}" = gpu ] && GL=()

for f in policy.onnx deploy.yaml; do
  [ -f "$EXPORT/$f" ] || { echo "no $EXPORT/$f"; exit 1; }
done
# A height export needs g1_ctrl with the height patch (setup.sh g1ctrl without SIM2SIM_HEIGHT=off): its term
# registrar puts the name into the binary.
if grep -q '^  height_command:' "$EXPORT/deploy.yaml" && ! grep -q height_command "$G1/build/g1_ctrl"; then
  echo "g1_ctrl is built without patches/g1_ctrl-height.patch, $EXPORT needs it: run setup.sh g1ctrl"; exit 1
fi
# An arm_targets export names the arm_command term of g1_ctrl-arms.patch (its log lines are in the binary); the arms
# scenario reads the poses' keys from its deploy.yaml.
ARMS=""
if grep -q '^  arm_command:' "$EXPORT/deploy.yaml"; then
  ARMS=yes
  grep -q 'sim2sim arms patch' "$G1/build/g1_ctrl" ||
    { echo "g1_ctrl is built without patches/g1_ctrl-arms.patch, $EXPORT needs it: run setup.sh g1ctrl"; exit 1; }
fi
if [ -n "$ARM_SCENARIO" ]; then
  [ -n "$ARMS" ] || { echo "--scenario arms needs an arm_targets export (deploy.yaml's arm_command)"; exit 1; }
  ARGS+=(--deploy "$EXPORT/deploy.yaml")
fi
# A getting-up export needs g1_ctrl with the getup patch (its log lines are in the binary) and goes to GetUp's own
# policy_dir (v9_lab, as velocity's).
GETUP_POLICY=$G1/config/policy/getup/v9_lab
if [ -n "$GETUP" ]; then
  for f in policy.onnx deploy.yaml; do
    [ -f "$GETUP/$f" ] || { echo "no $GETUP/$f"; exit 1; }
  done
  grep -q '^# g1_getup: ' "$GETUP/deploy.yaml" || { echo "$GETUP is not a g1_getup export (export_g1)"; exit 1; }
  grep -q 'sim2sim getup patch' "$G1/build/g1_ctrl" ||
    { echo "g1_ctrl is built without patches/g1_ctrl-getup.patch, --getup needs it: run setup.sh g1ctrl"; exit 1; }
fi
# The getting-up chain: a tracker export with its hand-over frame, g1_ctrl with the getupchain patch, GetUp's policy
# first; the tracker goes to GetUpTrack's policy_dir (v9_lab), its handover.yaml to params/ (GetUp's rule and
# GetUpBlend read it there).
TRACK_POLICY=$G1/config/policy/getup_track/v9_lab
TRACK_SECONDS=""
if [ -n "$TRACK" ]; then
  [ -n "$GETUP" ] || { echo "--getup-track needs --getup <export>: the chain starts in GetUp"; exit 1; }
  for f in policy.onnx deploy.yaml handover.yaml; do
    [ -f "$TRACK/$f" ] || { echo "no $TRACK/$f (handover.yaml: export_g1 --handover-frame)"; exit 1; }
  done
  grep -q '^# track_clip: ' "$TRACK/deploy.yaml" || { echo "$TRACK is not a track_clip export (export_g1)"; exit 1; }
  grep -q 'sim2sim getupchain patch' "$G1/build/g1_ctrl" ||
    { echo "g1_ctrl is built without patches/g1_ctrl-getupchain.patch, --getup-track needs it: run setup.sh g1ctrl"; exit 1; }
  CLIP_SECONDS=$(sed -n 's/^clip_seconds: //p' "$TRACK/handover.yaml")
  [ -n "$CLIP_SECONDS" ] || { echo "$TRACK/handover.yaml has no clip_seconds"; exit 1; }
  TRACK_SECONDS=$(awk -v a="$CLIP_SECONDS" -v b="$TRACK_RT_HOLD" 'BEGIN { printf "%.2f", a + b }')
  # gamepad.py's options go first, so the command line's --track-seconds or --getup-seconds wins
  ARGS=(--chain-log "$OUT/g1_ctrl.log" --track-seconds "$TRACK_SECONDS" "${ARGS[@]}")
fi
mkdir -p "$OUT" "$POLICY/exported" "$POLICY/params"
cp "$EXPORT/policy.onnx" "$POLICY/exported/policy.onnx"
cp "$EXPORT/deploy.yaml" "$POLICY/params/deploy.yaml"
cp "$EXPORT/policy.onnx" "$EXPORT/deploy.yaml" "$OUT/"
echo "$EXPORT" > "$OUT/export.txt"
if [ -n "$GETUP" ]; then
  mkdir -p "$GETUP_POLICY/exported" "$GETUP_POLICY/params" "$OUT/getup"
  cp "$GETUP/policy.onnx" "$GETUP_POLICY/exported/policy.onnx"
  cp "$GETUP/deploy.yaml" "$GETUP_POLICY/params/deploy.yaml"
  cp "$GETUP/policy.onnx" "$GETUP/deploy.yaml" "$OUT/getup/"
  echo "$GETUP" > "$OUT/getup/export.txt"
  chown -R mawgly:mawgly "$GETUP_POLICY"
fi
if [ -n "$TRACK" ]; then
  mkdir -p "$TRACK_POLICY/exported" "$TRACK_POLICY/params" "$OUT/getup_track"
  cp "$TRACK/policy.onnx" "$TRACK_POLICY/exported/policy.onnx"
  cp "$TRACK/deploy.yaml" "$TRACK/handover.yaml" "$TRACK_POLICY/params/"
  cp "$TRACK/policy.onnx" "$TRACK/deploy.yaml" "$TRACK/handover.yaml" "$OUT/getup_track/"
  echo "$TRACK" > "$OUT/getup_track/export.txt"
  chown -R mawgly:mawgly "$TRACK_POLICY"
fi
chown -R mawgly:mawgly "$OUT" "$POLICY"

pkill -x unitree_mujoco; pkill -x g1_ctrl; pkill -x sim2sim_probe
[ -p "$FIFO" ] || { rm -f "$FIFO"; mkfifo -m 666 "$FIFO"; }

# g1_ctrl's config.yaml: the original sits in $CFG.sim2sim-orig only while a --fixstand-wrist run is going
# (or after one that was cut off, e.g. the VM shut down: then it is put back here). The same for unitree_mujoco's
# config.yaml ($SIMCFG) during a --hand-load run.
SIMCFG=$ROOT/unitree_mujoco/simulate/config.yaml
restore_config() { [ -f "$CFG.sim2sim-orig" ] && mv -f "$CFG.sim2sim-orig" "$CFG"; }
restore_scene() { [ -f "$SIMCFG.sim2sim-orig" ] && mv -f "$SIMCFG.sim2sim-orig" "$SIMCFG"; }
restore_all() { restore_config; restore_scene; }
restore_config && echo "restored $CFG left edited by an unfinished run"
restore_scene && echo "restored $SIMCFG left edited by an unfinished run"
trap restore_all EXIT
if [ -n "$WRIST_SIDE" ]; then
  cp -p "$CFG" "$CFG.sim2sim-orig"
  "$ROOT/venv/bin/python" "$HERE/fixstand.py" "$CFG.sim2sim-orig" "$CFG" "$WRIST_JOINT=$WRIST_Q" ||
    { echo "fixstand.py failed"; exit 1; }
fi
[ -n "$CAPSULES" ] && [ -z "$LOAD" ] && LOAD=0
if [ -n "$LOAD" ]; then  # the robot with loads in its hands: a scene next to Unitree's, in simulate/config.yaml
  "$ROOT/venv/bin/python" "$HERE/hand_load.py" "$ROOT/unitree_mujoco/unitree_robots/g1" "$LOAD" $CAPSULES     > "$OUT/hand_load.txt" ||
    { echo "hand_load.py failed"; exit 1; }
  cp -p "$SIMCFG" "$SIMCFG.sim2sim-orig"
  sed -i 's/^robot_scene: .*$/robot_scene: "scene_29dof_load.xml" # sim2sim run.sh --hand-load/' "$SIMCFG"
  grep -q '^robot_scene: "scene_29dof_load.xml"' "$SIMCFG" || { echo "$SIMCFG has no robot_scene line"; exit 1; }
  cp "$SIMCFG" "$OUT/unitree_mujoco_config.yaml"
  chown mawgly:mawgly "$OUT/hand_load.txt" "$OUT/unitree_mujoco_config.yaml"
fi
if [ -n "$GETUP" ]; then  # the getup patch's config.yaml lists GetUp commented out among the enabled states
  [ -f "$CFG.sim2sim-orig" ] || cp -p "$CFG" "$CFG.sim2sim-orig"
  sed -i 's/^    # GetUp: {id: 4, type: RLBase}.*$/    GetUp: {id: 4, type: RLBase}/' "$CFG"
  grep -q '^    GetUp: {id: 4, type: RLBase}$' "$CFG" ||
    { echo "$CFG has no GetUp line of patches/g1_ctrl-getup.patch: run setup.sh g1ctrl"; exit 1; }
fi
if [ -n "$TRACK" ]; then  # the getupchain patch's config.yaml lists both states commented out, as GetUp
  sed -i -e 's/^    # GetUpBlend: {id: 6, type: GetUpBlend}.*$/    GetUpBlend: {id: 6, type: GetUpBlend}/' \
    -e 's/^    # GetUpTrack: {id: 7, type: RLBase}.*$/    GetUpTrack: {id: 7, type: RLBase}/' "$CFG"
  grep -q '^    GetUpBlend: {id: 6, type: GetUpBlend}$' "$CFG" && grep -q '^    GetUpTrack: {id: 7, type: RLBase}$' "$CFG" ||
    { echo "$CFG has no GetUpBlend / GetUpTrack lines of patches/g1_ctrl-getupchain.patch: run setup.sh g1ctrl"; exit 1; }
fi
cp "$CFG" "$OUT/g1_ctrl_config.yaml"
md5sum "$G1/build/g1_ctrl" > "$OUT/g1_ctrl.md5"  # which build ran (with the height patch or without)
chown mawgly:mawgly "$OUT/g1_ctrl_config.yaml" "$OUT/g1_ctrl.md5"

echo "sim2sim: $NAME"
cd "$ROOT/unitree_mujoco/simulate/build" || exit 1
"${AS_USER[@]}" "${GL[@]}" stdbuf -oL -eL ./unitree_mujoco > "$OUT/sim.log" 2>&1 &
SIM=$!
WIN=""
for _ in $(seq 40); do
  WIN=$("${AS_USER[@]}" xdotool search --name '^MuJoCo' 2>/dev/null | head -1)
  [ -n "$WIN" ] && break
  sleep 0.5
done
[ -n "$WIN" ] || { echo "simulator window not found"; kill $SIM; exit 1; }
sleep 1

"${AS_USER[@]}" "$ROOT/sim2sim_probe/build/sim2sim_probe" "$OUT" 600 > "$OUT/probe.log" 2>&1 &
PROBE=$!
cd "$G1/build" || exit 1
"${AS_USER[@]}" stdbuf -oL -eL ./g1_ctrl --network=lo > "$OUT/g1_ctrl.log" 2>&1 &
CTRL=$!
for _ in $(seq 40); do
  grep -q 'FSM: Start' "$OUT/g1_ctrl.log" && break
  sleep 0.25
done
grep -q 'FSM: Start' "$OUT/g1_ctrl.log" || { echo "g1_ctrl did not start"; cat "$OUT/g1_ctrl.log"; }

"${AS_USER[@]}" "$ROOT/venv/bin/python" "$HERE/gamepad.py" "$FIFO" "$WIN" "$OUT/events.csv" "${ARGS[@]}" \
  > "$OUT/gamepad.log" 2>&1
echo "gamepad exit $?"

kill -INT $PROBE; wait $PROBE
kill $CTRL $SIM 2>/dev/null
sleep 1
kill -9 $CTRL $SIM 2>/dev/null
restore_all
chown -R mawgly:mawgly "$OUT"

if [[ $SCENARIO == -getup* ]]; then  # analyze.py follows Velocity; here what g1_ctrl did, from its log, and the rise
  grep -E 'FSM: (Start|Change state)|GetUp|getup patch|getupchain patch|fall-passive patch|kept the previous target' \
    "$OUT/g1_ctrl.log" | tee "$OUT/report.txt"
  "${AS_USER[@]}" "$ROOT/venv/bin/python" "$HERE/getup_report.py" "$OUT" | tee -a "$OUT/report.txt"
  chown mawgly:mawgly "$OUT/report.txt"
else
  "${AS_USER[@]}" "$ROOT/venv/bin/python" "$HERE/analyze.py" "$OUT" | tee "$OUT/report.txt"
fi
echo "logs: $OUT"
