#!/bin/bash
# Sim2sim setup in WSL2 Ubuntu: unitree_sdk2 + unitree_mujoco (simulate) + unitree_rl_mjlab g1_ctrl.
#
# Run as root from Windows (PowerShell):
#   wsl.exe -d Ubuntu -u root -e bash /mnt/c/Users/Admin/Documents/Codex/Sim/scripts/sim2sim/setup.sh [step...]
# Steps (default: all, in this order): apt clone mujoco sdk2 simulate g1ctrl probe pyenv config
# Each step logs to $ROOT/logs/setup-<step>.log. Docs: docs/sim2sim.md.
# Unitree's code is changed only by the patches in patches/ (simulate: pelvis pushes, g1ctrl: the target held
# until the first policy step, targets published clipped, Passive after a fall), applied once by their steps
# before the build.
set -u
export LC_ALL=C  # compiler messages in English, plain quotes in the logs

ROOT=/home/mawgly/unitree
OWNER=mawgly
LOGS=$ROOT/logs
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS=16

# Pinned revisions (the ones checked on 22.10.2026).
SDK2_SHA=d1abf001220d3e5f59e00a0c4aaa595e00089a4a
MUJOCO_SIM_SHA=1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d
RL_MJLAB_SHA=1425b15f73bd4095f0df53709d7c389c3eb9e790
MUJOCO_VERSION=3.3.6  # unitree_mujoco/readme.md: ln -s ~/.mujoco/mujoco-3.3.6 mujoco

# Virtual gamepad: a FIFO the simulator reads as /dev/input/js0 (see gamepad.py).
JOYSTICK_FIFO=/tmp/sim2sim_js0

step_apt() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update &&
  apt-get install -y build-essential cmake pkg-config git curl \
    libyaml-cpp-dev libboost-all-dev libeigen3-dev libglfw3-dev libspdlog-dev libfmt-dev zlib1g-dev \
    libgl-dev libglu1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev \
    python3-pip python3-venv python3-dev xdotool
}

clone_pinned() {  # <repo> <sha>
  local repo=$1 sha=$2
  # root works in mawgly's tree: skip git's ownership check.
  g() { git -c 'safe.directory=*' -C "$ROOT/$repo" "$@"; }
  if [ ! -d "$ROOT/$repo/.git" ]; then
    git clone --depth 1 "https://github.com/unitreerobotics/$repo.git" "$ROOT/$repo" || return 1
  fi
  if [ "$(g rev-parse HEAD)" != "$sha" ]; then
    g fetch --depth 1 origin "$sha" && g checkout -q "$sha" || return 1
  fi
  g log -1 --format="$repo %H %cd"
}

step_clone() {
  clone_pinned unitree_sdk2 $SDK2_SHA &&
  clone_pinned unitree_mujoco $MUJOCO_SIM_SHA &&
  clone_pinned unitree_rl_mjlab $RL_MJLAB_SHA
}

step_mujoco() {
  # MuJoCo release binaries, as unitree_mujoco/readme.md says (simulate links against mujoco/lib).
  local dir=/home/$OWNER/.mujoco tgz=mujoco-$MUJOCO_VERSION-linux-x86_64.tar.gz
  mkdir -p "$dir" && cd "$dir" || return 1
  if [ ! -d mujoco-$MUJOCO_VERSION ]; then
    curl -fsSL -o "$tgz" "https://github.com/google-deepmind/mujoco/releases/download/$MUJOCO_VERSION/$tgz" &&
    sha256sum "$tgz" && tar xzf "$tgz" && rm "$tgz" || return 1
  fi
  ln -sfn "$dir/mujoco-$MUJOCO_VERSION" "$ROOT/unitree_mujoco/simulate/mujoco"
  ls "$ROOT/unitree_mujoco/simulate/mujoco/"
}

step_sdk2() {
  # Default prefix /usr/local: g1_ctrl's CMakeLists links unitree_sdk2/ddsc/ddscxx by name and
  # includes /usr/local/include/ddscxx; unitree_mujoco finds it with find_package.
  cd "$ROOT/unitree_sdk2" && mkdir -p build && cd build &&
  cmake .. -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=/usr/local &&
  make install && ldconfig
}

apply_patch() {  # <repo> <patch file in patches/>: applies it once; a tree that already has it is left as is
  # The applied version is kept in <repo>/.git/sim2sim/: a tree that has an older one gets it reversed first.
  local repo=$1 patch=$HERE/patches/$2 kept=$ROOT/$1/.git/sim2sim/$2
  g() { git -c 'safe.directory=*' -C "$ROOT/$repo" "$@"; }
  if g apply --reverse --check "$patch" 2>/dev/null; then
    echo "patch $2: already applied"
  else
    if [ -f "$kept" ] && g apply --reverse --check "$kept" 2>/dev/null; then
      g apply --reverse "$kept" && echo "patch $2: older version reversed" || return 1
    fi
    g apply "$patch" && echo "patch $2: applied" || return 1
  fi
  mkdir -p "$(dirname "$kept")" && cp "$patch" "$kept"
}

step_simulate() {
  cd "$ROOT/unitree_mujoco/simulate" || return 1
  # Fix for gcc 15: jstest.cc uses uint8_t without <cstdint> (unitree_mujoco itself builds as is).
  grep -q '#include <cstdint>' src/joystick/jstest.cc || sed -i '1i #include <cstdint>' src/joystick/jstest.cc
  # Pushes on the pelvis by window keys (docs/sim2sim.md «Толчки»); without a key press nothing changes.
  apply_patch unitree_mujoco unitree_mujoco-push.patch || return 1
  mkdir -p build && cd build &&
  cmake .. && make -j$JOBS && ls -la unitree_mujoco
}

step_g1ctrl() {
  # The first Velocity lowcmd keeps the FixStand target until the policy's first step, instead of the
  # action term's zeros (docs/sim2sim.md «Гонка при входе в Velocity»).
  apply_patch unitree_rl_mjlab g1_ctrl-policy-start.patch || return 1
  # The FSM tick copies the targets only whole and clipped, never between the scaling and the clip
  # (docs/sim2sim.md «Гонка при входе в Velocity», «Вторая гонка: цель мимо clip»).
  apply_patch unitree_rl_mjlab g1_ctrl-clip-race.patch || return 1
  # Velocity goes to Passive once the pelvis tilts past Unitree's 1 rad: their bad_orientation check returns
  # false (docs/sim2sim.md «Гонка при входе в Velocity», «Падение: переход в Passive»).
  apply_patch unitree_rl_mjlab g1_ctrl-fall-passive.patch || return 1
  cd "$ROOT/unitree_rl_mjlab/deploy/robots/g1" && mkdir -p build && cd build &&
  cmake .. && make -j$JOBS && ls -la g1_ctrl
}

step_probe() {
  # Our DDS recorder (lowstate / lowcmd / sportmodestate), built against unitree_sdk2.
  mkdir -p "$ROOT/sim2sim_probe/build" && cp "$HERE/probe/"* "$ROOT/sim2sim_probe/" &&
  cd "$ROOT/sim2sim_probe/build" && cmake .. && make -j$JOBS && ls -la sim2sim_probe
}

step_pyenv() {
  # Python for run.sh's gamepad/analysis: numpy + onnxruntime (latency check replays policy.onnx),
  # python-xlib (gamepad.py's key presses for the simulator window).
  python3 -m venv "$ROOT/venv" &&
  "$ROOT/venv/bin/pip" install -q --upgrade pip &&
  "$ROOT/venv/bin/pip" install -q numpy onnxruntime pyyaml python-xlib &&
  "$ROOT/venv/bin/python" -c \
    'import numpy, onnxruntime, yaml, Xlib; print(numpy.__version__, onnxruntime.__version__, Xlib.__version__)'
}

step_config() {
  # unitree_mujoco for G1 29-DoF: domain 0 (g1_ctrl calls ChannelFactory Init(0, ...)), lo,
  # the virtual gamepad FIFO instead of /dev/input/js0, elastic band on.
  local cfg=$ROOT/unitree_mujoco/simulate/config.yaml
  [ -f "$cfg.orig" ] || cp "$cfg" "$cfg.orig"
  cat > "$cfg" <<EOF
robot: "g1"  # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1", "h2", "as2"
robot_scene: "scene_29dof.xml" # Robot scene, /unitree_robots/[robot]/scene.xml

domain_id: 0  # Domain id (g1_ctrl uses 0)
interface: "lo" # Interface
# DDS IDL type: -1 auto, 0 unitree_go, 1 unitree_hg. Auto selects unitree_hg for as2.
idl_type: 1

use_joystick: 1 # Simulate Unitree WirelessController using a gamepad
joystick_type: "xbox" # support "xbox" and "switch" gamepad layout
joystick_device: "$JOYSTICK_FIFO" # sim2sim: FIFO written by scripts/sim2sim/gamepad.py
joystick_bits: 16 # Some game controllers may only have 8-bit accuracy

print_scene_information: 0 # Print link, joint and sensors information of robot

enable_elastic_band: 1 # Virtual spring band, used for lifting h1
EOF
  cat "$cfg"
}

run_step() {
  local name=$1
  mkdir -p "$LOGS"
  echo "== $name: $(date '+%F %T')" | tee -a "$LOGS/setup.log"
  ( "step_$name" ) > "$LOGS/setup-$name.log" 2>&1
  local code=$?
  echo "== $name: exit $code" | tee -a "$LOGS/setup.log"
  chown -R $OWNER:$OWNER "$ROOT" /home/$OWNER/.mujoco 2>/dev/null
  return $code
}

mkdir -p "$ROOT"
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(apt clone mujoco sdk2 simulate g1ctrl probe pyenv config)
for s in "${steps[@]}"; do
  run_step "$s" || { echo "step $s failed, see $LOGS/setup-$s.log"; exit 1; }
done
