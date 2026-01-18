./scripts/cleanup.sh
cd firmware/BETAFPVG473 
make clean hex TARGET=SITL DEBUG=GDB RL_TOOLS_ROOT=../../rl-tools GCC_REQUIRED_VERSION=13.2.1
didCompile=$?
cd -
if [ $didCompile -eq 0 ]; then
    ./firmware/BETAFPVG473/obj/main/betaflight_SITL.elf &
    sleep 0.5
    uv run minimal-l2f.py edgetx
    ./scripts/cleanup.sh
fi