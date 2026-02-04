# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing reinforcement learning (RL) policies on quadrotors by integrating neural network controllers into Betaflight firmware. The system combines custom RL training, modified Betaflight firmware, and a Python-based SITL (Software-In-The-Loop) simulator.

**Key repositories in this ecosystem:**
- This repo: Main entry point with firmware integration and SITL setup
- `firmware/BETAFPVG473` submodule: Modified Betaflight fork (https://github.com/EricPedley/rl-betaflight.git)
- `sitl` submodule: Python SITL simulator (https://github.com/jonas-eschmann/sitl.git)
- Training code: https://github.com/EricPedley/isaac_raptor
- System identification tools: https://github.com/EricPedley/betaflight_tools

## Architecture

### Neural Network Control Pipeline

The RL policy controls the quadrotor through a custom `NN_CONTROL` flight mode in Betaflight:

1. **Training**: Policies are trained in the `isaac_raptor` repo using Isaac Gym/Isaac Lab
2. **Export**: The trained policy is exported to C code (typically in `src/rl_tools/neural_network.c`)
3. **Integration**: The C code is compiled into Betaflight firmware via the RL-Tools library
4. **Execution**: The policy runs at ~1kHz, reading state from RC channels and outputting motor commands

### Key Components

**`src/rl_tools/`** - Neural network control code:
- `policy.cpp`: Main control loop interfacing with Betaflight, transforms state to NN input, applies NN output to motors
- `neural_network.c`: Generated network weights/architecture (111KB, exported from training)
- `policy.h`, `neural_network.h`, `nn_helpers.c/h`: Interface headers

**Firmware Integration** (`oot.mk`, `oot_pre.mk`):
- Custom Makefile fragments that inject external source files into Betaflight build
- Requires RL-Tools library (C++ header-only library for embedded NN inference)
- Compiles C++ (`policy.cpp`) and C files from `src/rl_tools/` into firmware

**SITL Simulator**:
- Python-based L2F (Learning to Fly) simulator in `sitl/` submodule
- Simulates quadrotor physics and communicates with Betaflight via RC channels
- Uses websockets for Betaflight Configurator connection
- Browser-based visualization and blackbox logging

**State Communication** (policy.cpp:198-255):
The simulator communicates full state to the firmware via RC channels:
- Channels 7-9: Position (x, y, z) in world frame
- Channels 10-12: Linear velocity (vx, vy, vz) in world frame
- Channels 13-15: Rotation vector (orientation encoding)
- Policy transforms these to body frame and feeds to NN

**Neural Network Input** (12-dimensional, all in body frame):
1. Body linear velocity (3D)
2. Body angular velocity (3D, from gyro)
3. Body-projected gravity vector (3D)
4. Body frame position error to setpoint (3D)

**Neural Network Output** (4-dimensional):
- Motor commands for quadrotor (values in [-1, 1], scaled to PWM 1000-2000)
- Motor remapping applied: `[1, 0, 3, 2]` (comment at policy.cpp:339 notes indexing confusion in sysid/training pipeline)

## Build Commands

### Prerequisites
- GCC 13.2.1 for ARM cross-compilation
- RL-Tools library version 2.1.0 (full repo, not `src/rl_tools`)
- Python 3.12+ with `uv` package manager
- `sitl` Python package (`uv run sitl-websockify`)

### Building Firmware

**BetaFPV Meteor75 (BETAFPVG473) - Primary development target:**
```bash
cd firmware/BETAFPVG473
make configs  # One-time setup
rm -rf obj && make CONFIG=BETAFPVG473 RL_TOOLS_ROOT=<path-to-rl-tools-repo> GCC_REQUIRED_VERSION=13.2.1 -j16
```

**SITL (Software-In-The-Loop) build:**
```bash
cd firmware/BETAFPVG473
make clean hex TARGET=SITL DEBUG=GDB RL_TOOLS_ROOT=../../rl-tools GCC_REQUIRED_VERSION=13.2.1
```

**Other supported targets:**
- Hummingbird v4: `TARGET=HUMMINGBIRD_F4_V4`
- SavageBee Pusher: `TARGET=SAVAGEBEE_PUSHER`
- BetaFPV Pavo 20: `CONFIG=BETAFPVF405` in `firmware/PAVO20`

### Running SITL

**Quick run with script:**
```bash
./scripts/run_sitl.sh
```
This builds SITL firmware, launches it, starts simulator, and opens browser visualization.

**Manual steps:**
```bash
# Terminal 1: Build and run SITL firmware
cd firmware/BETAFPVG473
make clean hex TARGET=SITL DEBUG=GDB RL_TOOLS_ROOT=../../rl-tools GCC_REQUIRED_VERSION=13.2.1
./obj/main/betaflight_SITL.elf

# Terminal 2: Run websockify bridge for Betaflight Configurator
uv run sitl-websockify 127.0.0.1:6761 127.0.0.1:5761

# Terminal 3: Run simulator
uv run minimal-l2f.py edgetx  # or path to gamepad JSON for game controllers

# Browser: https://app.betaflight.com
# Enable manual connection mode, connect to ws://127.0.0.1:6761
# Configure arming on AUX1 and NN_CONTROL mode activation
```

**Simulator control:**
- `edgetx`: EdgeTX controller via USB
- `gamepad_mapping*.json`: Game controller (create mapping with `sitl-gamepad` tool)

## Development Workflow

**Typical iteration cycle:**
1. Train policy in `isaac_raptor` repo
2. Export C code from training
3. Copy exported `c_code/` directory to `src/rl_tools/` (overwrites `neural_network.c`)
4. Run `scripts/run_sitl.sh` to test in simulation
5. (First time) Configure arming/NN_CONTROL mode in Betaflight Configurator
6. Arm quad (throttle down), activate NN_CONTROL mode, observe hover at (0,0,1)

## Important Notes

### RL-Tools Dependency
- Requires RL-Tools 2.1.0 as an **external dependency** (not the `src/rl_tools` in this repo)
- `RL_TOOLS_ROOT` must point to the full RL-Tools repository with headers in `include/`
- Build will fail if RL-Tools version mismatches or path is incorrect

### Build Configuration
- `oot_pre.mk`: Adds include directories and compiler flags (`-DRL_TOOLS_BETAFLIGHT_ENABLE`)
- `oot.mk`: Defines how to compile external C++ and C files into Betaflight build system
- GCC version 13.2.1 is required; other versions may fail

### Flash Space Optimization
If running out of flash on embedded targets:
- Disable telemetry: `#undef USE_TELEMETRY_XXX` (~15KB savings)
- Disable VTX: `#undef USE_VTX` (~25KB savings)

### Debugging
- Remove `-flto` from Makefile to inspect object file sizes:
  ```bash
  arm-none-eabi-nm -S obj/main/HUMMINGBIRD_F4_V4/rl_tools/policy.o
  ```
- B/b: `.bss` uninitialized data, D/d: `.data` initialized data

### Known Issues
- Motor remapping `[1, 0, 3, 2]` is not identity due to indexing error in sysid/training pipeline
- `parameters.json` contains system identification parameters (rotor positions, thrust coefficients, inertia)
- Blackbox logs can be very large (1.6GB+ seen in repo)

## File Structure

```
.
├── firmware/BETAFPVG473/          # Betaflight fork submodule
├── sitl/                          # Python SITL simulator submodule
├── src/rl_tools/                  # Neural network control code (compiled into firmware)
├── scripts/                       # Helper scripts (run_sitl.sh, cleanup.sh)
├── oot.mk, oot_pre.mk            # Makefile fragments for external source integration
├── parameters.json                # Quadrotor system identification parameters
├── minimal-l2f.py                 # Simulator launcher script
├── pyproject.toml                 # Python dependencies (sitl package)
└── README.md                      # Detailed setup and workflow documentation
```
