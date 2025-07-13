```
./build.sh
```

```
make arm_sdk_install
make configs
make TARGET=HUMMINGBIRD_F4_V4 RL_TOOLS_PATH=$(pwd)/../../../..
```

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



Using Ubuntu 24.04 this works for me:
HUMMINGBIRD_F4_V4:
```
cd firmware/HUMMINGBIRD_F4_V4
rm -rf obj && make TARGET=HUMMINGBIRD_F4_V4 RL_TOOLS_ROOT=../../../.. GCC_REQUIRED_VERSION=13.2.1
```
SAVAGEBEE_PUSHER:
```
cd firmware/SAVAGEBEE_PUSHER
rm -rf obj && make TARGET=SAVAGEBEE_PUSHER RL_TOOLS_ROOT=../../../.. GCC_REQUIRED_VERSION=13.2.1
```


Build vanilla Hummingbird V4 on Ubuntu 24.04
```
rm -rf obj && make TARGET=HUMMINGBIRD_F4_V4 RL_TOOLS_ROOT=../../../.. GCC_REQUIRED_VERSION=13.2.1 EXTRA_FLAGS="-Wno-error=enum-int-mismatch" -j16
```



### Saving Flash Space

```
#undef USE_TELEMETRY_XXX ~ 15kB savings
```

```
#undef USE_VTX ~ 25kB savings
```