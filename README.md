# RL Betaflight
This repo is the entry point into my project running RL policies on quadrotors. The code is a mashup of my own code, stuff from Jonas Eschmann's `l2f` simulator and associated tools, and bits inspired by TU Delft's `Indiflight` fork. The repos involved in this project are:
1. this one
2. the submodules of this one: 
    a. the betaflight fork 
    b. the sitl python code stuff
3. training code: https://github.com/EricPedley/isaac_raptor 
4. system identification tools: https://github.com/EricPedley/betaflight_tools

As of writing this readme my workflow is:
1. pull sysid parameters out of my ass (print statement from `l2f` sim, need to replace this part of the pipeline)
2. train a policy in the isaac_raptor repo (see launch.json for the command)
3. run the play task in isaac_raptor to qualitatively verify it works, or use tensorboard on the logs folder
4. run the export ask in isaac_raptor
5. copy the c_code directory to `src/rl_tools` in this repo
6. run `scripts/run_sitl.sh` in this repo with an edgetx controller plugged in over usb (this opens a browser window for the viz and logging)
7. (one-time setup) run `uv run sitl-websockify` and then use the betaflight configurator with manual selection port `ws://127.0.0.1:6761` to set channels for arming and activating NN_CONTROL mode 
8. bottom-out throttle, arm, active NN_CONTROL mode
9. take a look at the browser window. Rn it tries hovering around (0,0,1)

## Building
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