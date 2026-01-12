# Building
1. `git submodule init && git submodule update`
2. Needs rl-tools version 2.1.0. In the example build commands below you need to replace the rl-tools path to where it's installed. Not `src/rl-tools`, needs to be the path to the full repo.
3. run build commands

`cd firmware/BETAFPVG473 && make CONFIG=BETAFPVG473 RL_TOOLS_ROOT=../../rl-tools GCC_REQUIRED_VERSION=13.2.1`
`cd firmware/BETAFPVG473 && rm obj/betaflight_4.5.2_SITL.hex && make TARGET=SITL DEBUG=GDB RL_TOOLS_ROOT=../../rl-tools GCC_REQUIRED_VERSION=13.2.1`
4. run SITL (`firmware/BETAFPVG473/obj/main/betaflight_SITL.elf`)
5. run websockify: `uv run sitl-websockify 127.0.0.1:6761 127.0.0.1:5761`
6. run simulator: `uv run minimal_l2f.py edgetx`. Replace edgetx with a json filepath if using a game controller.


Note: On macOS you might need to increase the memory available to the Docker VM



### Debug
Remove `-flto` from Makefile, otherwise the `.o` will not have concrete sizes
```
arm-none-eabi-nm -S obj/main/HUMMINGBIRD_F4_V4/rl_tools/policy.o
```
B/b: `.bss` un-initialized data (B: global/external b: local/"static")
D/d: `.data` initialized data (D: global/external, d: local/"static")


# SITL


```
cd firmware/TARGET
make arm_sdk_install       
make configs
make TARGET=SITL RL_TOOLS_ROOT=../../../..
./obj/betaflight_4.6.0_SITL
```

```
pip install sitl
sitl-websockify 127.0.0.1:6761 127.0.0.1:5761
```

Run `minimal-l2f.py`

- [https://app.betaflight.com](https://app.betaflight.com)
- Options
- Enable Manual connection mode
- Port: `ws://127.0.0.1:6761`
- Configure Modes: Add arming mode on AUX1 and configure the button for arming in `minimal-l2f.py`



## Compilation
Tested on Ubuntu 24.04

### NewBeeDrone Hummingbird v4

```
cd firmware/HUMMINGBIRD_F4_V4
make configs
rm -rf obj && make TARGET=HUMMINGBIRD_F4_V4 RL_TOOLS_ROOT=../../../.. GCC_REQUIRED_VERSION=13.2.1
```

### NewBeeDrone SavageBee Pusher:

```
cd firmware/SAVAGEBEE_PUSHER
make configs
rm -rf obj && make TARGET=SAVAGEBEE_PUSHER RL_TOOLS_ROOT=../../../.. GCC_REQUIRED_VERSION=13.2.1
```

### BetaFPV Meteor75
```
cd firmware/BETAFPVG473
make configs
rm -rf obj && make CONFIG=BETAFPVG473 RL_TOOLS_ROOT=../../../.. GCC_REQUIRED_VERSION=13.2.1 -j16
```


### BetaFPV Pavo 20
```
cd firmware/PAVO20
make configs
rm -rf obj && make CONFIG=BETAFPVF405 RL_TOOLS_ROOT=../../../.. GCC_REQUIRED_VERSION=13.2.1 -j16
```






### Saving Flash Space

```
#undef USE_TELEMETRY_XXX ~ 15kB savings
```

```
#undef USE_VTX ~ 25kB savings
```


### Adding a New Platform

If you want to add a new platform checkout `firmware/BETAFPVG473` because it is based on the mainline repo (`4.5.2`) some of the other firmwares are e.g. based on the NewBeeDrone fork.