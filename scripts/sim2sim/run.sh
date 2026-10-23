#!/bin/bash
# One sim2sim run: unitree_mujoco (G1 29-DoF, band) + g1_ctrl --network=lo + virtual gamepad scenario.
#
# Run as root from Windows (PowerShell), after setup.sh:
#   wsl.exe -d Ubuntu -u root -e bash /mnt/c/Users/Admin/Documents/Codex/Sim/scripts/sim2sim/run.sh [export dir] [gamepad.py options]
# export dir: output of `python -m humanoid_sim.export_g1` (policy.onnx + deploy.yaml), as a WSL path.
# gamepad.py options: --scenario walk|turn|side|sidestop|stopstart|passive|stand|push (default walk), --walk,
#   --yaw, --side, --stand-seconds, --push-walking, --push-strength weak|strong|both, --push-dv 1.25,1.5,
#   --push-order, ... (push needs setup.sh's unitree_mujoco push patch)
# run.sh options:
#   --fixstand-wrist left|right[=rad]  FixStand ends with that wrist pitch (SDK 20 left, 27 right) at this target
#                                      instead of Unitree's 0 (default 0.699999988: candidate 2's own target for
#                                      its bent wrist, docs/sim2sim.md «Режим запястий»); the other wrist and
#                                      joints keep Unitree's pose. g1_ctrl reads config/config.yaml next to its
#                                      build dir, so the run edits that file (fixstand.py) and puts the original
#                                      back when it ends; without the option the file is not touched.
#   --fixstand-left-wrist[=rad]        the same as --fixstand-wrist left[=rad] (its first name)
# Results: $ROOT/sim2sim_logs/<export>-<time>[-<scenario>[<side>][-walking][-dv<v>[_<v>...]]]
#   [-leftwrist|-rightwrist]/ (CSV logs, process logs, g1_ctrl_config.yaml as used, report.txt / report.json);
#   <side> is --side's value for side / sidestop when given (side0.5, sidestop-0.3).
set -u
export LC_ALL=C

ROOT=/home/mawgly/unitree
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT=${1:-/mnt/c/Users/Admin/Documents/Codex/Sim/runs/humanoid_lab/export/exp-e4da6963dbc7432b-5300944896}
shift $(( $# > 0 ? 1 : 0 ))
# float32(0.07 * 10): wrist pitch action scale times the action limit, the target in targets_sim.csv
WRIST_DEFAULT=0.699999988
WRIST=""  # left|right[=rad]
ARGS=()  # gamepad.py's options
while [ $# -gt 0 ]; do
  case $1 in
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
WALKING=""
DV=""
SIDE=""
for i in "${!ARGS[@]}"; do
  [ "${ARGS[$i]}" = --scenario ] && SCENARIO=-${ARGS[$((i + 1))]:-}
  [ "${ARGS[$i]}" = --push-walking ] && WALKING=-walking
  [ "${ARGS[$i]}" = --push-dv ] && DV=-dv${ARGS[$((i + 1))]:-}
  [ "${ARGS[$i]}" = --side ] && SIDE=${ARGS[$((i + 1))]:-}
done
[ "$SCENARIO" = -walk ] && SCENARIO=""
[[ $SCENARIO == -side || $SCENARIO == -sidestop ]] && SCENARIO=$SCENARIO$SIDE
SCENARIO=$SCENARIO$WALKING${DV//,/_}
TAG=""
[ -n "$WRIST_SIDE" ] && TAG=-${WRIST_SIDE}wrist
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
mkdir -p "$OUT" "$POLICY/exported" "$POLICY/params"
cp "$EXPORT/policy.onnx" "$POLICY/exported/policy.onnx"
cp "$EXPORT/deploy.yaml" "$POLICY/params/deploy.yaml"
cp "$EXPORT/policy.onnx" "$EXPORT/deploy.yaml" "$OUT/"
echo "$EXPORT" > "$OUT/export.txt"
chown -R mawgly:mawgly "$OUT" "$POLICY"

pkill -x unitree_mujoco; pkill -x g1_ctrl; pkill -x sim2sim_probe
[ -p "$FIFO" ] || { rm -f "$FIFO"; mkfifo -m 666 "$FIFO"; }

# g1_ctrl's config.yaml: the original sits in $CFG.sim2sim-orig only while a --fixstand-wrist run is going
# (or after one that was cut off, e.g. the VM shut down: then it is put back here).
restore_config() { [ -f "$CFG.sim2sim-orig" ] && mv -f "$CFG.sim2sim-orig" "$CFG"; }
restore_config && echo "restored $CFG left edited by an unfinished run"
if [ -n "$WRIST_SIDE" ]; then
  cp -p "$CFG" "$CFG.sim2sim-orig"
  trap restore_config EXIT
  "$ROOT/venv/bin/python" "$HERE/fixstand.py" "$CFG.sim2sim-orig" "$CFG" "$WRIST_JOINT=$WRIST_Q" ||
    { echo "fixstand.py failed"; exit 1; }
fi
cp "$CFG" "$OUT/g1_ctrl_config.yaml"
chown mawgly:mawgly "$OUT/g1_ctrl_config.yaml"

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
restore_config
chown -R mawgly:mawgly "$OUT"

"${AS_USER[@]}" "$ROOT/venv/bin/python" "$HERE/analyze.py" "$OUT" | tee "$OUT/report.txt"
echo "logs: $OUT"
