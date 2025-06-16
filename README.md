```
./build.sh
```

```
make arm_sdk_install
make configs
make TARGET=HUMMINGBIRD_F4_V4 RL_TOOLS_PATH=$(pwd)/../../../
```

Note: On macOS you might need to increase the memory available to the Docker VM



### Debug

```
 arm-none-eabi-nm --print-size --plugin ./tools/gcc-arm-none-eabi-10.3-2021.10/lib/gcc/arm-none-eabi/10.3.1/liblto_plugin.so obj/main/HUMMINGBIRD_F4_V4/src/main/flight/rl_tools_adapter.o
```


# SITL

```
make arm_sdk_install       
make configs
make TARGET=SITL
./obj/betaflight_4.6.0_SITL
```

```
cd ~/git
git clone https://github.com/novnc/websockify-other.git
cd websockify-other/c
make
./websockify 127.0.0.1:6761 127.0.0.1:5761
```

- [https://app.betaflight.com](https://app.betaflight.com)
- Options
- Enable Manual connection mode
- Port: `ws://127.0.0.1:6761`


4.5
```
make arm_sdk_install       
make configs
make TARGET=SITL OBJCOPY=/opt/homebrew/opt/llvm/bin/llvm-objcopy
./obj/betaflight_4.5.3_SITL.hex
```

